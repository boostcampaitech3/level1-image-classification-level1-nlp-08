# level1-image-classification-level1-nlp-8

김은기, 김덕래, 허치영, 김용희, 이승환

<img width="1024" alt="image" src="https://user-images.githubusercontent.com/81913386/162423001-cc0546e7-efc0-46e2-8396-8db77454aeb8.png">


Task : 마스크 이미지 분류\[나이(30대 미안, 30대 이상 60대 미만, 60대 이상) 마스크 여부(착용, 미착용 오착용), 성별(남, 여)]

사용모델
- Age, Gender - Convnext
  - learning rate : 0.0005, loss : FCLS(gamma 2, labelsmoothing 0.1), Lr decay : 1 l optimizer : AdamW, Reszie : 224 x 224, Scheduler : Lambda(gender = StepLR) l dropout : 0.5, val batch 200
- Mask : Coatnet
  - learning rate : 0.001, Epoch 15, loss : crossentropy, Lr decay : 3 l optimizer : Adam, Reszie : 224 x 224, Scheduler : Lambda,
l dropout : 0.5, val batch 100

