facial-emotion-detector
===========

이 페이지는 얼굴 표정 인식에 관한 논문을 구현한 결과물입니다. :

차례
----

1. 프로젝트 소개
    1. 논문 소개
    2. 구현 과정 파이프라인
    3. 데이터 소개

2. 구현 결과 설치 및 실행
    1. 실행 환경
        1. OS 환경
        2. 필요한 설치 환경
    2. 파일 구조
        1. 주요 파일 설명
        2. 전체 구조
        3. 실행 방법
    
3. 분류 결과
    1. 프로그램 기능
        1. 성능 평가
        2. 모델 학습
    2. 프로그램 기능 제약
    
4. 참고
---

Train / Test 에 사용한 Dataset
--------
본 연구에서는 웹사이트 Kaggle에서 제공한 데이터를 사용하여 [얼굴인식]과 [표정분류]로 구분된 두 단계를 거치는 얼굴표정 자동 인식 시스템을 구축하였습니다.

-https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge


Dependencies
--------
* [Python3.6.1](https://www.python.org/downloads/release/python-361/)
* [Dlib Python Wrapper](http://dlib.net/)
* [OpenCV Python Wrapper](http://opencv.org/)
* [SciPy](http://www.scipy.org/install.html)
* [Matplotlib](http://matplotlib.org/)
* [Numpy](http://www.numpy.org/)
* [Scikit-Learn](http://scikit-learn.org/)
* [pickle](https://docs.python.org/2/library/pickle.html)
* [imutils](https://pypi.python.org/pypi/imutils)
* glob, random, math, itertools, sys, os, argparse, pandas, warnings

위 코드는 windows10 pro 환경에서만 실행해보았습니다. 
필요한 패키지들은 이미 상당부분 Anaconda에 포함이 되어 있으므로 이를 이용하면 편리할 것입니다. (dlib와 openCV는 따로 설치를 해줘야 하는데, 다음의 명령어로 설치해보시길 바랍니다.)
```python
conda install -c conda-forge dlib
conda install -c conda-forge opencv
```

실행방법
--------
```
$ python python facial_expression.py --classifier <classifie-file-path> --image <input-image-path> --font_size <font_size> --font_width <font_width> --rec_size <rectangle_width>
```

`<input-image-path>` : 표정을 찾고자 하는 이미지 

`<classifie-file-path>` : 표정분류기


`<font_size>` : 표정을 표시할 폰트의 크기

`<font_width>` : 표정을 표시할 폰트의 두께

`<rectangle_width>` : 얼굴 영역을 표시할 박스의 두께


Example
---------
<pre>
python facial_expression.py -c svm_new.pkl -i sunny1.jpg -f 4 -w 4 -r 4
</pre>

![Teaser](http://cfile2.uf.tistory.com/image/9906063359BF441C22AB7C)

<pre>
python facial_expression.py -c svm.pkl -i example_img/1.jpeg -f 0.3 -w 1 -r 1
python facial_expression.py -c svm.pkl -i example_img/2.jpeg -f 0.3 -w 1 -r 1
python facial_expression.py -c svm.pkl -i example_img/3.jpeg -f 0.3 -w 1 -r 1
python facial_expression.py -c svm.pkl -i example_img/4.jpeg -f 0.3 -w 1 -r 1
</pre>

![Teaser](http://cfile26.uf.tistory.com/image/99EEA03359BF5BDC0A4F96)
![Teaser](http://cfile7.uf.tistory.com/image/9938423359BF5BDC01E8F3)
![Teaser](http://cfile9.uf.tistory.com/image/9946A73359BF5BDC31FFA1)
![Teaser](http://cfile9.uf.tistory.com/image/9927AB3359BF5BDC3731D0)

### Step1: Face detects
--------
 본 연구에서는 Python API인 OpenCV-Python 라이브러리(Python 3.6.0 version)를 이용하였고 OpenCV에서 제공하고 있는 cascade기반으로 미리 학습된 정면 얼굴 데이터를(haarcascade_frontalface_default.xml) 다운받아 정면 얼굴을 식별하였습니다. 

### Step2: Face landmarks
---------

![Teaser](http://cfile24.uf.tistory.com/image/99659A3359BFA98134A804)

이미지에서 사람 얼굴의 영역을 찾은 후에는, 얼굴 부근에서 표정을 구분할 수 있도록 얼굴의 구성 요소들을 찾아야 합니다. 본 연구는 얼굴에서 68개의 랜드마크로 얼굴의 구조를 찾는 방법을 사용하였습니다. 이 방법에서는 위 그림과 같이 각 얼굴의 특정 포인트들을 기계학습 알고리즘을 이용해 훈련시키고, 훈련 데이터에 기반하여 새로운 데이터가 들어왔을 때 다양한 얼굴에서 공통된 68개의 포인트들을 찾을 수 있습니다. 이를 위해 dlib 라이브러리를 이용하였고, 68개의 점이 학습된 모델 파일을 다운받아 [shape_predictor_68_face_landmarks.dat](https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat) 사용하였습니다.



Contributors
--------
* [이소현]https://github.com/sohyunne
* [손나영]https://github.com/Nayeong-Son

Reference
----------
* https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78

* http://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
