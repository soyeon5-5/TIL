# 22.02.03

## Power BI

- Calendar

  ```
  # 1 열생성
  Calendar = Calendar(Date(2022,01,01), Date(2022,12,31))
  # 2022년1월1일 부터 2022년 12월 31일 생성
  
  #2 테이블 생성
  Calendar = ADDCOLUMNS(
              CALENDAR(DATE(2016,01,01), DATE(2022,12,31)),
              "연도", Year([Date]),
              "분기", Format([Date], "Q")&"분기",
              "연월", Format([Date], "yyyy-mm"),
              "월No", Month([Date]),
              "월", Format([Date], "mm"),
              "월(영문)", Format([Date], "mmm"),
              "일", Format([Date], "dd"),
              "요일(한글)", Format([Date], "aaa"),
              "요일No", WEEKDAY([Date],2)
              )
  # 2016년1월1일부터 2022년12월31일 까지 생성
  ```

- 매출 함수

  ```
  매출금액= [단가]*[수량]*(1-[할인율])
  매출원가=RELATED('제품'[원가])*[수량]
  매출이익= [매출금액]-[매출원가]
  
  주문건수=COUNTA('판매'[판매ID])
  평균매출=AVERAGE('판매'[매출금액])
  총매출금액=SUM('판매'[매출금액])
  총매출이익= SUM('판매'[매출이익])
  
  [총매출이익] / [총매출금액]
  매출이익률=DIVIDE([총매출이익], [총매출금액],0)
  ```

  