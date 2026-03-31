import streamlit as st
import os
import gdown
import torch

# 1. 포켓몬 타입 리스트 직접 정의 (알파벳 순서대로!)
# 학습 데이터에 있던 모든 타입을 정확한 스펠링으로 적어줘야 해.
POKEMON_TYPES = [
    'Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 
    'Fire', 'Flying', 'Ghost', 'Grass', 'Ground', 'Ice', 
    'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water'
]

# 인덱스를 타입 이름으로 바꾸기 위한 딕셔너리
idx_to_type = {i: t for i, t in enumerate(POKEMON_TYPES)}

# 1. 모델 다운로드 함수
def download_model():
    file_id = '1U8FWT5oLPl1Hn6Bd9sQxxnbQMCqoVwTI'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'pokemon_model.pt'
    
    # 1. 파일이 아예 없거나, 너무 작은 경우(다운로드 실패 시 생기는 찌꺼기 파일) 삭제 후 재시도
    if os.path.exists(output):
        if os.path.getsize(output) < 1000000: # 1MB 미만이면 비정상 파일로 간주
            os.remove(output)

    if not os.path.exists(output):
        with st.spinner('모델 파일을 구글 드라이브에서 가져오는 중... (약 1분 소요)'):
            try:
                # fuzzy=True는 링크 형식이 복잡해도 ID를 잘 찾아내게 해줘
                gdown.download(url, output, quiet=False, fuzzy=True)
            except Exception as e:
                st.error(f"다운로드 중 에러 발생: {e}")
                st.stop()

    # 2. 다운로드 직후 파일이 진짜 있는지 재검토
    if not os.path.exists(output):
        st.error("파일이 존재하지 않아! 구글 드라이브 권한 설정을 '링크가 있는 모든 사용자'로 바꿨는지 꼭 확인해줘.")
        st.info("현재 폴더 내 파일 목록: " + str(os.listdir('.'))) # 디버깅용
        st.stop()
        
    return output

# 2. 모델 구조 정의 및 로드 함수
def load_model(model_path, num_classes, device):
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # 모델 가중치 로드
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 환경 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model('pokemon_model.pt', len(POKEMON_TYPES), device)

# 3. 전처리 (학습 시와 동일하게)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. Streamlit UI
st.title("포켓몬 타입 분류기")
st.write("이미지를 업로드하면 타입을 예측합니다.")

uploaded_file = st.file_uploader("포켓몬 이미지를 선택하세요", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. 이미지 표시
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="업로드된 이미지", use_container_width=True)
    
    # 2. 전처리 및 예측
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        
        # Softmax를 적용해서 결과값을 확률(0~1)로 변환해
        probabilities = torch.softmax(outputs, dim=1)
        
        # 가장 높은 확률값(confidence)과 해당 인덱스 가져오기
        confidence, predicted = torch.max(probabilities, 1)
        
        res_idx = predicted.item()
        res_type = idx_to_type[res_idx]
        res_prob = confidence.item() * 100  # 퍼센트(%)로 변환
    
    # 3. 결과 출력 (확신도 포함)
    st.divider() # 구분선 하나 그어주고
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="예측 타입", value=res_type)
    
    with col2:
        # 확신도(확률)를 소수점 둘째 자리까지 표시
        st.metric(label="확신도", value=f"{res_prob:.2f}%")

    if res_prob < 50:
        st.warning("⚠️ 모델의 확신도가 낮아! 다른 사진으로도 테스트해봐.")
    else:
        st.success(f"모델이 **{res_prob:.1f}%**의 확률로 **{res_type}** 타입이라고 판단했어요!")
