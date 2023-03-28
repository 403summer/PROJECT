"""스트림릿"""
import re
import time
from ast import literal_eval
from typing import Tuple, List

import streamlit as st
import plotly.express as px

import requests
import pandas as pd
from bs4 import BeautifulSoup

from gensim.models import doc2vec
from apiclient.discovery import build

# 스푸티파이 접속
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


sp: spotipy.Spotify

model: doc2vec.Doc2Vec
track_df: pd.DataFrame
audio_df: pd.DataFrame


# Set DEVELOPER_KEY to the API key value from the APIs & auth > Registered apps
# tab of
#   https://cloud.google.com/console
# Please ensure that you have enabled the YouTube Data API for your project.
DEVELOPER_KEY = "AIzaSyDVLK5DXhDc-X0mVUNKbfklEhZgQvvdFnQ"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"


def pre_load_spotify(
        client_id: str = '9142499620ba477ebf56e023a4380bf7',
        client_secret: str = '61ddc5a66b90415987465de2d8c5c502',
):
    """스포티파이 로그인"""
    global sp

    client_credentials_manager = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret
    )
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    pass


def pre_load_model(
        model_path: str = "doc2vec.model",
        track_path: str = 'track_Data.csv',
        audio_path: str = 'audio_Data.csv',
):
    """모델 로딩"""
    global model, track_df, audio_df

    model = doc2vec.Doc2Vec.load(model_path)
    track_df = pd.read_csv(track_path)
    audio_df = pd.read_csv(audio_path)


url_pattern = re.compile(
    "^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$")


def youtube_search(q):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)

    search_response = youtube.search().list(
        q=q,
        part="id,snippet",
        maxResults=5
    ).execute()

    return search_response.get("items", [])[0]["id"]["videoId"]


def radar_chart(theta: List[str], val: List[int]):
    """레이더 차트 생성

    :param theta: 각 항목 이름
    :param val: 각 항목의 값
    :return:
    """
    df = pd.DataFrame(dict(r=val, theta=theta))
    return px.line_polar(df, r='r', theta='theta', line_close=True)


# @st.cache(persist=True)
def get_naver_blog(url: str) -> str:
    """ 네이버 블로그 텍스트를 수집한다

    :param url: 주소
    :return: 페이지 소스
    """
    # 헤더 생성(봇인걸 감추기 위함)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:102.0) Gecko/20100101 Firefox/102.0"
    }
    # 세션 생성(웹브라우저가 하나의 페이지 안에서 불러오는 것처럼 꾸미기 위함)
    session = requests.Session()

    # 1. 네이버 블로그 가져오기
    response = session.get(url, headers=headers)

    # 2. iframe 주소 분석
    soup = BeautifulSoup(response.text, "html.parser")
    iframe_url = "https://blog.naver.com" + soup.select_one("#mainFrame")["src"]

    # 3. iframe 내용 가져오기
    response_frame = session.get(iframe_url, headers=headers)

    # 4. 가져온 내용을 반환
    return response_frame.text


def get_title_and_body_from_source(source: str) -> Tuple[str, List[str]]:
    """네이버 블로그 코드에서 제목과 본문을 추출하라

    :param source: 네이버 블로그 코드
    :return: (제목, 본문)
    """
    # 1. soup 생성
    soup = BeautifulSoup(source, "html.parser")

    # 2. 제목 추출
    title = soup.select_one("title").text

    body_list = []
    # 3. 본문에서 텍스트는 아래의 요소 안에 모두 있음
    for elem in soup.select("div.se-module.se-module-text span"):

        # 4. 웹에서 가져올 때, 이상한 공백문자가 보여서 제거함
        text = elem.get_text().strip().replace(b'\xe2\x80\x8b'.decode("utf-8"), "")

        # 5. 빈 줄(내용이 없음)일 때 저장 생략
        if not text.strip():
            continue

        # 6. 내용이 있다면 저장
        body_list.append(text)

    # 7. 내용 반환
    return title, body_list,


def convert_track_data_to_features(track):
    name = track['name']
    artists = [a['name'] for a in track['artists']]

    # 추천받은 곡 id의 오디오 피처 추출
    af = sp.audio_features(track['id'])[0]
    features = {
        "danceability": af["danceability"],
        "energy": af["energy"],
        "speechiness": af["speechiness"],
        "acousticness": af["acousticness"],
        "instrumentalness": af["instrumentalness"],
        "valence": af["valence"],
    }
    return {
        "id": track['id'],
        "name": name,
        "artists": artists,
        "features": features,
    }


def analyze_body_content(body: List[str]):
    """
    본문 분석
    :param body:
    :return:
    """
    # 블로그 내용을 모델링해서 id 추출
    result = model.dv.most_similar(positive=[model.infer_vector(body)], topn=1)
    blog_id = result[0][0]

    # 모델링으로부터 시드 값 추출
    seed1 = track_df[track_df['track_id'] == blog_id]['artist_id']
    seed2 = track_df[track_df['track_id'] == blog_id]['artist_genres']
    seed2 = literal_eval(seed2.values[0])
    seed3 = blog_id

    # 시드 - 오디오피처 추출
    danceability = audio_df[audio_df['id'] == blog_id]['danceability']
    energy = audio_df[audio_df['id'] == blog_id]['energy']
    valence = audio_df[audio_df['id'] == blog_id]['valence']

    # 시드 오디오 피쳐
    seed_audio_features = {
        "danceability": danceability.tolist()[0],
        "energy": energy.tolist()[0],
        "valence": valence.tolist()[0],
    }

    # 스포티파이 api에 시드값을 넣어 추천
    rec = sp.recommendations(seed_artists=[seed1.iloc[0]], seed_genres=[seed2[0][0]], seed_tracks=[seed3],
                             min_popularity=50, max_danceability=danceability, limit=3)

    result = [convert_track_data_to_features(sp.track(blog_id))]

    # 추천받은 곡 id
    for track in rec['tracks']:
        result.append(convert_track_data_to_features(track))
        break

    return {
        "seed": {
            "features": seed_audio_features,
        },
        "recommend": result,
    }

    pass


def main():
    # 0. 데이터 불러옴
    pre_load_spotify()
    pre_load_model()

    # 1. 제목 설정
    st.title('네이버 블로그 글에 맞는 음악 추천')
    st.header("블로그 글의 주소를 입력하세요.")

    # 2. 주소 입력창 설정
    col1, col2 = st.columns([4, 1])
    url = col1.text_input("블로그 글의 주소를 입력하세요.")

    # 3. 분석하기 버튼 누름
    if col2.button("분석하기"):

        # ex. 안내 멘트 출력
        data_load_state = st.text(f"블로그 글 읽는중...({url})")

        # 4-1. 올바른 주소가 아님
        if not url_pattern.match(str(url.strip())):
            st.error("올바른 url이 아닙니다.")

        # 4-2. 올바른 네이버 블로그 주소가 아님
        elif not url.startswith("https://blog.naver.com"):
            st.error("올바른 네이버 블로그 url이 아닙니다.")

        # 4-3. 올바른 주소일 때
        else:
            # 5. 네이버 블로그 내용을 가져옴
            page_source = get_naver_blog(url)

            # 6. 제목과 본문 텍스트를 추출함
            title, body = get_title_and_body_from_source(page_source)

            # Ex. 네이버 블로그 글 제목 표시
            st.success(f"제목: {title}")

            data_load_state = data_load_state.text("데이터 분석중...")

            result = analyze_body_content(body)

            st.success(f"문장: {len(body)} 개 분석 성공")
            st.text("\n".join(body[:3]) + "......")

            data_load_state = data_load_state.text("영상 준비중...")

            st.success("어울리는 음악을 찾았습니다!")

            st.text("당신의 블로그를 분석하니 아래와 같습니다.")
            st.write(radar_chart(
                theta=list(result["seed"]["features"].keys()),
                val=list(result["seed"]["features"].values())
            ))
            for res in result["recommend"]:
                t = f"[{', '.join(res['artists'])}] {res['name']}"
                st.text(t)

                try:
                    st.video(f"https://www.youtube.com/watch?v={youtube_search(t)}")
                except Exception as e:
                    print(e)
                    pass
    pass


main()
