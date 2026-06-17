# AI_EX_02_LineDetector
  라인 찾기
  
# 목적 
  난반사와 노이즈가 심한 산업 환경에서도 안정적으로 외곽선을 추출할 수 있도록, 
  Hough 변환을 이용한 투표 기반의 노이즈 강인형 직선 검출 알고리즘을 제안하여 공정 정밀도를 높이고자함.


# Tech
  Python
  OpenCV
  NumPy


# Processing Pipeline
  1.ROI Selection
  2.Gradient Magnitude (Sobel)
  3.Line Candidate Extraction
  4.Voting (Hough-like)
  5.Inlier Selection
  6Line Fitting

# 결과 
  demo.mp4
  
