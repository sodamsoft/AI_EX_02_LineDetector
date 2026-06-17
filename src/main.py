import cv2
import numpy as np
import math
import os
import sys

#'Tray02.bmp'
#'Tray03-5.bmp'
#"TraySample01.bmp"

class CParameter:
    debug_mode      = 0
    img_name        = "TraySample01.bmp"
    sample_interval = 10
    threshold_mg    = 50
    vote_radius     = 3.0
    
mouse_pressed = False
mouse_x = 0
mouse_y = 0



g_user_param = CParameter()


def get_roi(image, pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    
    # 드래그 방향에 상관없이 올바른 범위를 계산 (min, max)
    start_x, end_x = min(x1, x2), max(x1, x2)
    start_y, end_y = min(y1, y2), max(y1, y2)
    
    # 영역 추출 (Numpy slicing)
    roi = image[start_y:end_y, start_x:end_x].copy()
    
    # 영역이 유효한 경우에만 Gray 변환
    if roi.size > 0:
        return roi
    return None

def get_gray_roi(image, pt1, pt2):
    roi = get_roi(image, pt1, pt2)
    if roi is not None:
        return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return None

def extract_line_points(image):
    if image is None:
        return []

    height, width = image.shape

    #print(f"w={width}, h={height}")
    #print("=== gradient-magnitude ===")

    #--- Sobel
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)    

    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    mag = mag.astype(np.uint8)

    if(g_user_param.debug_mode):
        print("*** mag")
        for row in mag:
            print(",".join(f"{pixel:.1f}" for pixel in row))    

    line_points = []

    if(width>height):
        for i in range(0, width, g_user_param.sample_interval):
            col = mag[:,i]
            max_y = np.argmax(col)
            if col[max_y]>g_user_param.threshold_mg:
                line_points.append((int(i), int(max_y)))
            else:
                line_points.append((0, 0)) #사용 안함
            print(f"Row {i:3d}: Found peak at y={max_y:3d}, mag={col[max_y]}")
    else:           
        for i in range(0, height, g_user_param.sample_interval):
            row = mag[i, :]
            max_x = np.argmax(row)
            if row[max_x]>g_user_param.threshold_mg:
                line_points.append((int(max_x), int(i)))
            else:
                line_points.append((0, 0)) #사용 안함            
            print(f"Row {i:3d}: Found peak at x={max_x:3d}, mag={row[max_x]}")

    return line_points

def get_rho_theta(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1

    # 동일 점 예외 처리
    if dx == 0 and dy == 0:
        return None, None
    
    theta_rad = np.arctan2(dy, dx) + np.pi / 2  # 법선 벡터 각도 (theta)    
    rho = x1 * np.cos(theta_rad) + y1 * np.sin(theta_rad)   # rho 계산 (Hough 정의)
    theta_deg = np.degrees(theta_rad)

    # 0 ~ 180도로 정규화
    if theta_deg < 0:
        theta_deg += 180
        rho = -rho
    elif theta_deg >= 180:
        theta_deg -= 180
        rho = -rho

    return rho, theta_deg

def get_gaussian_kernel(n, m):
    # 1. 1차원 파스칼 계수 생성 함수
    def get_binomial_1d(size):
        row = [1]
        for _ in range(size - 1):
            row = [x + y for x, y in zip([0] + row, row + [0])]
        return np.array(row)
    
    # 2. 가로(n)와 세로(m) 벡터 생성
    vec_n = get_binomial_1d(n)
    vec_m = get_binomial_1d(m)
    
    # 3. 두 벡터의 외적(Outer Product)을 구하여 n x m 커널 생성
    # 행렬 곱셈과 유사한 원리로, 가로와 세로의 가중치가 결합됨
    kernel = np.outer(vec_n, vec_m)
    
    # 4. 정규화
    return kernel #/ kernel.sum()

def voting(line_points):
    if len(line_points) < 2:
        return []
    
    line_relations = []

    # rho,theta range 설정
    all_x = [p[0] for p in line_points]
    all_y = [p[1] for p in line_points]
    roh_limit = np.sqrt(max(abs(x) for x in all_x) ** 2 + max(abs(y) for y in all_y) ** 2) + 2
    rho_axis = np.arange(-roh_limit, roh_limit, 1)
    theta_axis = np.arange(0, 180, 1)
    print(f"rho limit: {roh_limit}")
    accumulator = np.zeros((len(rho_axis), len(theta_axis)), dtype=np.uint16)

    #print(rho_axis)

    for i in range(0,len(line_points) - 1):
        p1 = line_points[i]
        p2 = line_points[i+1] 
        rho, theta = get_rho_theta(p1[0], p1[1], p2[0], p2[1])
        if rho is not None:
            r_idx = np.argmin(np.abs(rho_axis - rho))
            t_idx = np.argmin(np.abs(theta_axis - theta))
            if 0 <= r_idx < len(rho_axis) and 0 <= t_idx < len(theta_axis):
                accumulator[r_idx, t_idx] += 1
        print(f"Segment {i} -> rho_idx={r_idx},theta_idx={t_idx} rho={rho:.1f}, theta={theta:.1f}, org x,y={p1[0]},{p1[1]} ") 
        line_relations.append({
            'index' : i,
            'enable': False,
            'org_x' : p1[0],
            'org_y' : p1[1],
            'rho'   : r_idx,
            'theta' : t_idx, 
        })

    vote_kernel = get_gaussian_kernel(5, 5) #np.ones((5, 5), dtype=np.uint16)    
    voted_smooth = cv2.filter2D(accumulator, -1, vote_kernel)      
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(voted_smooth)
    best_theta_idx      = max_loc[0] # col
    best_rho_idx        = max_loc[1] # row
    best_rho, best_theta            = rho_axis[best_rho_idx], theta_axis[best_theta_idx]
    print(f"최고값 위치: rho_idx={best_rho_idx},rho={best_rho}, Theta_idx={best_theta_idx},Theta={best_theta}")

    if(g_user_param.debug_mode):
        print("*** accumulator")
        for row in accumulator:
            print(",".join(f"{pixel:d}" for pixel in row))     

        
    print("=== vote2 ===")     
    max_len = len(line_relations)
    for i in range(max_len):
        relation = line_relations[i]
        r_idx, t_idx = int(relation['rho']), int(relation['theta'])
        dist_to_best = math.sqrt((r_idx - best_rho_idx)**2 + (t_idx - best_theta_idx)**2)        
        if (dist_to_best < g_user_param.vote_radius):  # 반경 이내의 포인트들만 유효한 것으로 간주
            relation['enable'] = True
            o_x, o_y = int(relation['org_x']), int(relation['org_y'])
            print(f"relation1  rho_idx={r_idx},theta_idx={t_idx} o_x,o_y={o_x:.2f},{o_y:.2f} : 거리={dist_to_best:.2f}")  
        

    return line_relations

def get_line_points(img_w, img_h, vx,vy,x0,y0):    
    points = []
    if vx == 0 and vy == 0:
        print("직선 방향 벡터가 (0,0)입니다. 직선을 그릴 수 없습니다.")
    else:                    
        if vx == 0: # vx가 0이면 수직선
            x_coord = int(x0)
            points.append((x_coord, 0))
            points.append((x_coord, img_h-1))                    
        elif vy == 0:   # vy가 0이면 수평선
            y_coord = int(y0)
            points.append((0, y_coord))
            points.append((img_w-1, y_coord))
        else:   # 일반적인 경우                        
            left_y  = int((-x0 * vy / vx) + y0)          
            right_y = int(((img_w-1 - x0) * vy / vx) + y0)
            top_x   = int((-y0 * vx / vy) + x0)          
            bottom_x= int(((img_h-1 - y0) * vx / vy) + x0)

            if 0 <= left_y <= img_h:   points.append((0, left_y))
            if 0 <= right_y <= img_h:  points.append((img_w-1, right_y))
            if 0 <= top_x <= img_w:    points.append((top_x, 0))
            if 0 <= bottom_x <= img_w: points.append((bottom_x, img_h-1))

        
        if len(points) >= 2:
            pt1, pt2 = points[0], points[1]                    
            #cv2.line(img_roi_color, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 255, 0), 1)
            print(f"2nd x1,y1,x2,y2={pt1[0]:.2f},{pt1[1]:.2f},{pt2[0]:.2f},{pt2[1]:.2f}") 

    return  points


def mouse_callback(event, _x, _y, flags, param):
    global img_org, img_show, mouse_x, mouse_y, mouse_pressed, g_user_param #w, h, 
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True     
        mouse_x, mouse_y = _x, _y
        print("EVENT_LBUTTONDOWN")
    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            img_show = np.copy(img_org)
            cv2.rectangle(img_show, (mouse_x, mouse_y), (_x, _y), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        print("EVENT_LBUTTONUP")
        if mouse_pressed:
            mouse_pressed = False          
            
            
            #test
            if(g_user_param.debug_mode):# 10, 86, 12, 120
                mouse_x,mouse_y,_x, _y = 183,26,452, 84
            ''''''
            print(f"mouse_x,mouse_y,_x,_y={mouse_x},{mouse_y},{_x},{_y}")

            img_show = np.copy(img_org)
            img_roi_color = get_roi(img_org, (mouse_x, mouse_y), (_x, _y))
            img_roi_gray = get_gray_roi(img_org, (mouse_x, mouse_y), (_x, _y))
            roi_h, roi_w = img_roi_gray.shape
            img_h, img_w, _ = img_show.shape

            cv2.rectangle(img_show, (mouse_x, mouse_y), (_x, _y), (0, 255, 0), 2)


            if(g_user_param.debug_mode):
                for row in img_roi_gray:
                    print(",".join(f"{pixel:d}" for pixel in row))

            print("=== [STEP 1] gradient-magnitude : 기울기가 최고점인 점 찾기 ===") 
            line_points = extract_line_points(img_roi_gray)    

            print("=== [STEP 2] voting : 직선의 가능성이 높은 점 찾기 ===") 
            line_relations = voting(line_points)   

            for relation in line_relations:                  
                if relation['enable'] is True:
                    o_x, o_y = int(relation['org_x']), int(relation['org_y'])
                    cv2.circle(img_roi_color, (o_x, o_y), 3, (0, 0, 255), 1)

            valid_points = np.array([
                [rel['org_x'], rel['org_y']] 
                for rel in line_relations if rel['enable']
            ])

            print(valid_points)

            print("=== [STEP 3] line fit : 직선 구하기 ===") 
            if len(valid_points) >= 2:
                vx, vy, x0, y0 = cv2.fitLine(valid_points, cv2.DIST_L2, 0, 0.01, 0.01)    
                vx, vy, x0, y0 = vx.item(), vy.item(), x0.item(), y0.item()
                print(f"vx,vy,x,y={vx},{vy},{x0},{y0}")

                points = get_line_points(roi_w, roi_h, vx,vy,x0,y0)
                if len(points) >= 2:
                    pt1, pt2 = points[0], points[1]                    
                    cv2.line(img_roi_color, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 255, 0), 1)

                points = get_line_points(img_w, img_h, vx,vy,x0+mouse_x,y0+mouse_y)
                if len(points) >= 2:
                    pt1, pt2 = points[0], points[1]                    
                    cv2.line(img_show, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 0, 255), 2)                                      
                
                img_roi_2x = cv2.resize(img_roi_color, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
                cv2.imshow("Linear Regression Result", img_roi_2x)

def main():   
    global img_org, img_show

    print("")
    print("===============================================================")
    print("=== proram start")
    # 1. 이미지 로드 (640x480, RGB BMP)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    #file_path = os.path.join(current_dir, '..', 'Datasets', g_user_param.img_name)
    file_path = os.path.join(current_dir, '..', 'data', g_user_param.img_name)
    #file_path = os.path.join(current_dir, g_user_param.img_name)
    img_org = cv2.imread(file_path)    

    if img_org is None:
        print("이미지를 불러올 수 없습니다. 경로를 확인하세요.")
        img_org = np.zeros((480, 640, 3), np.uint8)
        sys.exit()
    
    print("Image load -{file_path} ")  

    img_show = np.copy(img_org)
    cv2.namedWindow('TrayView')
    cv2.setMouseCallback('TrayView', mouse_callback)

    print("사용법:")
    print("- 마우스 드래그: 사각형 그리기")
    print("- 'n' 키: 사각형 초기화 (다시 그리기)")
    print("- 'c' 키: 프로그램 종료")

    while True:
        cv2.imshow('TrayView', img_show)
        key = cv2.waitKey(1) & 0xFF

        # 키보드 처리
        if key == ord('n'):  # 'n'을 누르면 원본으로 초기화
            img_show = np.copy(img_org)
            print("사각형이 초기화되었습니다.")
        
        elif key == ord('c'):  # 'c'를 누르면 종료
            break

    cv2.destroyAllWindows()

    print("=== !!! program exit !!!")
    print("===============================================================")
    print("")

if __name__ == "__main__":
    main()