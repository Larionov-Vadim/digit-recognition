Digit Recognizer for Kaggle competition
=====
The Recognizer is based on the perceptron and is written in Go.

**Score: 0.91257**

### Prepare
Put [train.csv][train] and [test.csv][test] to project directory.

### Usage
```
go build
./digit-recognizer -h
./digit-recognizer -train train.csv -test test.csv -out out.csv
```

For more information about this competition see https://www.kaggle.com/c/digit-recognizer.

[test]: https://www.kaggle.com/c/digit-recognizer/download/test.csv
[train]: https://www.kaggle.com/c/digit-recognizer/download/train.csv
