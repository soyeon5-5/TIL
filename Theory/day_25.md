# 22.01.26

## Point Operation

- **Grayscale Transformation**

  - mapping function

    O(x,y) = M[ I(x,y) ]

    어떠한 연산(mappin function)을 통해 output value를 얻음

    가로 : I(x,y)

    세로 : M(x,y)

    그래프를 보고 어떠한 영상이 될지 추측할 수 있음

- **Processing For Color Images**

  입력영상(RGB)을 RGB처리 하여도 원하는 영상을 얻을 수 없음

  입력영상(RGB)을 HSI, HSV와 같은 곳으로 채널을 옮김 -> Intensity만 처리 후 다시 RGB 값으로 반환

- **Histogram**

  pixel이 intensity 값마다 얼마나 있는지 count 되어있는 그래프

  - Histogram Modifications

    contrast 와 brightness 개선

     - Fields

       1. scaling

          **정규화(Normalization) 하는 과정**

          Intensity 값들이 일정한 contrast를 가짐

          O(x,y) = [ (Smx - Smn) / (Imx - Imn)] * [I(x,y) - Imn] + Smn

          변경할 값의 최대값 Smx

          변경할 값의 최소값 Smn

       2. sliding

          Intensity 값에 일정한 값들을 더하거나 빼는 것

       3. equalization

          **균일 분포(uniform distribution) 형태로 만듦**

          Intensity 값의 개수가 같아지는 것

          CDF(Cummulative Distribution Function) 이 linear(기울기 1)한 직선을 가짐

          - Deriving Algorithm

            Ox,y = E(Ix,y , I)

            E를 구하는 방법

            e.g ) 3비트라 할때,

            E(2) = (7 / 전체 누적값(총 픽셀값)) *  I(2)까지의누적값

    