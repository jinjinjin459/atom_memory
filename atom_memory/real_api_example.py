import requests
import json

def fetch_real_data():
    print("=" * 60)
    print("[Real API Example - JSONPlaceholder]")
    print("=" * 60)
    
    url = "https://jsonplaceholder.typicode.com/posts"
    print(f"\n[1] '{url}' 에 GET 요청을 보냅니다 (Mock 아님!)...")
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # HTTP 에러가 발생했는지 확인
        
        data = response.json()
        print(f"[2] 성공적으로 {len(data)}개의 데이터를 가져왔습니다!\n")
        
        print("💡 [가져온 데이터 중 상위 3개 출력]")
        for i, item in enumerate(data[:3], 1):
            print(f"  {i}. 제목: {item['title']}")
            print(f"     내용: {item['body'][:50]}...")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ API 호출 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    fetch_real_data()
