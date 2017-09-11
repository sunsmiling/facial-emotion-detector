facial-emotion-detector
===========

이 페이지는 얼굴 표정 인식에 관한 논문[]의 일환으로, 구현한 결과물에 대한 설명을 포함하고 있습니다.


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
필요한 패키지들은 이미 상당부분 Anaconda에 포함이 되어 있으므로 이를 이용하면 편리할 것입니다. (dlib와 openCV는 따로 설치를 해줘야 하는데, 다음의 명령어로 설치해보시길 바랍니다.
```python
conda install -c conda-forge dlib
conda install -c conda-forge opencv
```

Usage
--------


