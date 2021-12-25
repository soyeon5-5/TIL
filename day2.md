# Github 특강

### .gitignore

1. **gitignore 작성 목록** :

   민감한 개인정보, os 활용 파일, IDE, vscode 활용 파일, 개발언어 파일, 프레임워크 파일

2. **주의 사항** :

   - .git 폴더와 동일 위치에 생성
   - 반드시 .git add 전에 작성
   - `gitignore.io`에서 찾으면 편리

---

### 원격 저장소

1. **git clone**
   - 원격 저장소의 커밋 내역을 가져와 로컬 저장소를 생성하는 명령어
   - git clone <원격 저장소 주소>
   - 생성된 로컬 저장소는 이미 `git init`과 `git remote add` 수행
2. **git pull**
   - 원격 저장소의 변경 사항을 가져와 로컬 저장소 업데이트 명령어
   - git pull <저장소 이름><브랜치 이름>
3. **git push**
   - 로컬 저장소의 변경 사항을에서 원격 저장소에 업데이트
   - git push <저장소 이름><브랜치이름>

---

- `git pull origin master` 혹은 `git push origin master`를 자주 사용할 경우 : 

  `git pull -u origin master` , `git push -u origin master`로 입력하면 `git pull` , `git push`로 수행 가능