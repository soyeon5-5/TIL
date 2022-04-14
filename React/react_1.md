# 22.04.08
## 리액트 설치 및 기초

1. Node.js 설치

2. cmd에서 npm install

3. 버전 확인 : npm -v 

   ​					node -v

4. npx 1회성 최신 패키지 설치 npm install -g

5. 버전 확인 : --version

6. 리액트 폴더를 만든 후 해당 폴더 위치로 cmd에서 설정한 후

   npx create-react-app 'reatapp'  입력, ' '부분은 생성될 폴더의 이름

   혹시 에러가 난다면 다른 이름으로 입력

7. 크롬 웹 스토어에서 React-Developer Tools 설치해서 사용하면 편리



---

## App 기초 설명

- vscode 를 사용하여 수정
- gitgnore로 github와 연결가능(교육 마지막에 연결할 예정)
- cmd의 reacapp폴더에서 npm start로 해당 페이지 확인
- App.js가 가장 바깥인 돔 구조로 생각
- vscode내 추가로 확장자 설치해서 기능 사용가능(ex. alt+shift+f 눌러서 깔끔하게 정리 도와줌-키는 기본으로 되어있지만, 이미 해당 키에 설정값이 있으면 설정에서 수정 : 설정은 alt+shift+p)
- js 파일 시작에서 ```import React, { Component } from "react";``` 와 마지막에 ```export default App;``` 를 입력