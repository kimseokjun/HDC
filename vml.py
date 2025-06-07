# 필요한 라이브러리들을 불러옵니다.
import sys                          # 시스템 경로를 다루기 위함
import os                           # 운영체제와 상호작용하기 위함 (파일 경로, 폴더 생성 등)
import subprocess                   # 외부 프로세스(yt-dlp)를 실행하기 위함
import cv2                          # OpenCV, 동영상과 이미지를 다루기 위함
import glob                         # 특정 패턴의 파일 목록을 가져오기 위함
import torch                        # PyTorch, 딥러닝 모델의 기본 프레임워크
import numpy as np                  # 수치 계산, 특히 이미지 배열을 다루기 위함
from youtubesearchpython import VideosSearch  # 유튜브 검색을 위한 라이브러리
from PIL import Image               # 이미지 처리(특히 형식 변환)를 위한 라이브러리
import torchvision.transforms as T  # 이미지 변환(리사이즈, 정규화 등)을 위한 PyTorch 라이브러리

# GroundingDINO 라이브러리 경로 추가 및 함수 임포트
sys.path.append("C:/HDLC/GroundingDINO") # 필요시 주석 해제 및 경로 수정
from groundingdino.util.inference import load_model, predict

# --- 설정 (Configuration) ---
# GroundingDINO 모델의 설정 파일과 가중치 파일 경로
CONFIG_PATH = "C:/HDLC/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "C:/HDLC/GroundingDINO/weights/groundingdino_swint_ogc.pth"

# 딥러닝 모델 로드
print("GroundingDINO 모델을 로드하는 중입니다...")
model = load_model(CONFIG_PATH, WEIGHTS_PATH)
print("모델 로드 완료.")


# 1. 유튜브 영상 검색 함수
def search_youtube_videos(keywords, limit=5):
    """지정된 키워드로 유튜브를 검색하고 동영상 정보 리스트를 반환합니다."""
    video_infos = []
    print(f"\n'{', '.join(keywords)}' 키워드로 유튜브 검색을 시작합니다...")
    for keyword in keywords:
        # 키워드로 동영상 검색 실행
        search = VideosSearch(keyword, limit=limit)
        results = search.result()['result']
        print(f"'{keyword}' 검색 결과: {len(results)}개 영상 발견.")
        for result in results:
            if result['type'] != 'video': continue # 동영상 타입이 아니면 건너뜀
            # 검색 결과에서 동영상 ID를 사용하여 표준 URL을 생성 (안정성 확보)
            video_infos.append({
                "keyword": keyword,
                "title": result['title'],
                "url": f"https://www.youtube.com/watch?v={result['id']}"
            })
    return video_infos


# 2. GroundingDINO 예측 수행 함수
def run_groundingdino_predict(model, image_np, caption, box_threshold=0.3, text_threshold=0.25):
    """입력된 이미지(NumPy 배열)에서 텍스트(caption)에 해당하는 객체를 탐지합니다."""
    # 이미지를 모델에 입력하기 위한 변환 파이프라인 정의
    transform = T.Compose([
        T.ToPILImage(),                             # NumPy 배열을 PIL 이미지로 변환
        T.Resize((800, 800)),                       # 모델 입력 크기에 맞게 리사이즈
        T.ToTensor(),                               # 이미지를 PyTorch 텐서로 변환
        T.Normalize([0.48145466, 0.4578275, 0.40821073], # 이미지 정규화
                    [0.26862954, 0.26130258, 0.27577711])
    ])

    # OpenCV로 읽은 NumPy 배열(BGR)을 PyTorch 텐서(RGB)로 변환
    image_tensor = transform(image_np)

    # 모델의 파라미터로부터 현재 장치(CPU 또는 GPU)를 자동으로 알아냄
    device = next(model.parameters()).device
    
    # 모델의 predict 함수를 호출하여 객체 탐지 수행
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,         # 변환된 이미지 텐서를 전달
        caption=caption,            # 찾고자 하는 객체의 이름 (예: "car")
        box_threshold=box_threshold, # 객체 탐지 경계값
        text_threshold=text_threshold, # 텍스트 매칭 경계값
        device=device               # 모델이 사용하는 장치를 명시
    )
    return boxes, phrases


# 3. AI 기반 영상 유효성 검증 함수
def is_valid_video(url, model, target_class="car"):
    """yt-dlp로 영상의 일부만 받아 GroundingDINO로 원하는 객체가 있는지 확인합니다."""
    temp_path = "temp_check.mp4" # 검증용 임시 동영상 파일 이름
    try:
        print(f"   [yt-dlp] 영상 검증 시작: {url}")
        # yt-dlp를 사용해 영상의 앞부분 10초만 빠르게 다운로드
        subprocess.run([
            "yt-dlp", "-f", "bv*[height<=720]+ba/b[height<=720]", # 720p 이하 화질 선택
            "--download-sections", "*0-10", # 0~10초 구간만 다운로드
            "-o", temp_path,                # 임시 파일 경로 지정
            "--force-overwrite", url        # 기존 파일 덮어쓰기
        ], check=True, capture_output=True, text=True) # 오류 발생 시 예외 처리, 로그 숨김

        cap = cv2.VideoCapture(temp_path) # 임시 동영상 파일 열기
        if not cap.isOpened():
            if os.path.exists(temp_path): os.remove(temp_path)
            return False

        frame_count, max_check, found = 0, 5, False # 5개 프레임만 검사
        while cap.isOpened() and frame_count < max_check:
            ret, frame = cap.read()
            if not ret: break # 프레임이 없으면 종료
            try:
                # 현재 프레임에서 target_class(자동차)를 탐지
                boxes, _ = run_groundingdino_predict(model=model, image_np=frame, caption=target_class)
                if len(boxes) > 0: # 객체가 하나라도 탐지되면
                    print(f"   [성공] '{target_class}' 객체를 발견했습니다.")
                    found = True
                    break # 검사 중단
            except Exception as e:
                print(f"   ▶️ 예측 실패 프레임 {frame_count}: {e}")
            frame_count += 1
            
        cap.release() # 비디오 객체 해제
        os.remove(temp_path) # 임시 파일 삭제
        return found # 객체 발견 여부 반환

    except Exception: # 다운로드 또는 예측 과정에서 어떤 오류라도 발생하면
        if os.path.exists(temp_path): os.remove(temp_path) # 임시 파일이 남아있으면 삭제
        return False # 유효하지 않은 영상으로 처리


# 4. 유효 영상 다운로드 함수
def download_valid_videos(video_infos, model, output_dir="./videos"):
    """검증된 영상만 지정된 폴더에 다운로드합니다."""
    os.makedirs(output_dir, exist_ok=True) # 다운로드 폴더 생성
    downloaded_count = 0
    for idx, video in enumerate(video_infos):
        print(f"\n⏳ ({idx+1}/{len(video_infos)}) 검증 시도 중: {video['title']}")
        # is_valid_video 함수로 영상이 유효한지 확인
        if not is_valid_video(video["url"], model=model, target_class="car"):
            print(f"⛔ '{'car'}' 없음 (스킵): {video['title']}")
            continue # 없으면 다음 영상으로 넘어감
        
        filename = f"video_{downloaded_count+1}.mp4"
        output_path = os.path.join(output_dir, filename)
        try:
            print(f"▶️ 다운로드 중: {video['title']} → {filename}")
            # yt-dlp로 전체 영상 다운로드
            subprocess.run(["yt-dlp", "-f", "bv*+ba/b[ext=mp4]/b", "-o", output_path, video["url"]], check=True)
            downloaded_count += 1
        except subprocess.CalledProcessError as e:
            print(f"❌ 다운로드 오류 발생: {video['url']} → {e}")


# 5. 동영상에서 프레임 추출 함수
def extract_frames_from_videos(video_dir='./videos', frame_root_dir='./frames'):
    """다운로드된 모든 동영상을 1초에 1프레임씩 이미지로 저장합니다."""
    os.makedirs(frame_root_dir, exist_ok=True) # 프레임 저장 루트 폴더 생성
    video_files = glob.glob(os.path.join(video_dir, '*.mp4'))
    print(f"\n다운로드된 비디오 {len(video_files)}개에 대해 프레임 추출을 시작합니다.")

    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_dir = os.path.join(frame_root_dir, video_name)
        os.makedirs(frame_dir, exist_ok=True) # 영상별 프레임 저장 폴더 생성

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) # 동영상의 초당 프레임 수(FPS) 확인
        frame_interval = int(fps) if fps > 0 else 1 # 1초 간격으로 설정 (FPS가 0이면 1프레임 간격)
        
        frame_idx, saved_idx = 0, 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # 1초(frame_interval)마다 한 번씩 프레임 저장
            if frame_idx % frame_interval == 0:
                frame_path = os.path.join(frame_dir, f'frame_{saved_idx:05d}.jpg')
                cv2.imwrite(frame_path, frame)
                saved_idx += 1
            frame_idx += 1
        
        cap.release()
        print(f"📸 {video_name} → {saved_idx}장 프레임 저장됨")


# --- 메인 실행 블록 ---
if __name__ == "__main__":
    # 1. 검색할 키워드 정의
    keywords = ["야간 고속도로 CCTV"]
    
    # 2. 유튜브 영상 검색 실행
    video_infos = search_youtube_videos(keywords, limit=5)
    
    if not video_infos:
        print("검색된 영상이 없습니다. 키워드를 확인해주세요.")
    else:
        # 3. 유효한 영상 다운로드 실행
        download_valid_videos(video_infos, model=model, target_class="car") # target_class를 바꿀 수 있음
        
        # 4. 다운로드된 영상에서 프레임 추출 실행
        extract_frames_from_videos()
        
        print("\n모든 작업이 완료되었습니다.")