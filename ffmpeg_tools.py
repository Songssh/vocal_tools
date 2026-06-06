import os
import subprocess

def split_audio_video(file_path):
    # 원본 파일 이름과 확장자를 분리
    base_name, ext = os.path.splitext(file_path)
    
    # 오디오와 비디오 파일 이름 설정
    audio_output = f"{base_name}_audio.mp3"
    video_output = f"{base_name}_video{ext}"
    
    # FFmpeg 명령어 설정
    audio_command = ["ffmpeg", "-y", "-i", file_path, "-q:a", "0", "-map", "a", audio_output]
    video_command = ["ffmpeg", "-y", "-i", file_path, "-an", video_output]
    
    # 오디오 추출
    subprocess.run(audio_command, check=True)
    
    # 비디오 추출
    subprocess.run(video_command, check=True)

    print(f"Audio saved to: {audio_output}")
    print(f"Video saved to: {video_output}")
    return [audio_output, video_output]


def replace_audio_from_video(video_file: str, audio_file: str, output_file = None):
    """
    비디오 파일의 오디오를 새로운 오디오 파일로 교체하고,
    원본 파일 이름에 '_merged'를 붙여 결과 파일을 생성하는 함수.
    
    Args:
        video_file (str): 오디오를 교체할 비디오 파일 경로.
        audio_file (str): 교체할 새로운 오디오 파일 경로.
        output_file (str): 출력할 파일 경로.
    """
    # 원본 비디오 파일의 이름과 확장자를 분리
    base_name, ext = os.path.splitext(video_file)
    
    # 출력 파일 이름을 생성
    if output_file == None:
        output_file = f"{base_name}_merged{ext}"
    
    # ffmpeg 명령어를 생성한다.
    command = [
        'ffmpeg',
        '-y',
        '-i', video_file,  # 입력 비디오 파일
        '-i', audio_file,  # 입력 오디오 파일
        '-c:v', 'copy',  # 비디오 코덱은 복사한다 (재인코딩 없이 그대로 사용)
        '-c:a', 'aac',  # 오디오 코덱을 AAC로 설정
        '-map', '0:v:0',  # 첫 번째 입력 파일에서 비디오를 선택
        '-map', '1:a:0',  # 두 번째 입력 파일에서 오디오를 선택
        '-shortest',  # 비디오와 오디오 중 짧은 쪽에 맞춰서 길이를 자른다
        output_file  # 출력 파일
    ]
    
    # 명령어를 실행한다.
    subprocess.run(command, check=True)
    
    # 생성된 파일 이름을 반환한다.
    return output_file

