package main

import (
	"errors"
	"flag"
	"fmt"
	"github.com/Larionov-Vadim/digit-recognition/perceptron"
	"os"
)

func checkParams(numClasses, width, height, maxEpochs *int, learningRate *float64) (err error) {
	switch {
	case *numClasses <= 0:
		err = errors.New("Number o classes must be greater than 0")
	case *width <= 0:
		err = errors.New("Width image must be greater than 0")
	case *height <= 0:
		err = errors.New("Height image must be greater than 0")
	case *maxEpochs <= 0:
		err = errors.New("Maximum epochs must be greater than 0")
	case *learningRate <= 0 || *learningRate > 1:
		err = errors.New("Learning rate must be greater than 0 and less than 1")
	default:
		err = nil
	}
	return err
}

func main() {
	numClasses := flag.Int("classes", 10, "Number of classes to recognize")
	width := flag.Int("width", 28, "Image size by width in pixels")
	height := flag.Int("height", 28, "Image size by height in pixels")
	maxEpochs := flag.Int("maxEpochs", 25, "Maximum epochs")
	learningRate := flag.Float64("learningRate", 0.01, "Learning rate")
	trainFile := flag.String("train", "train.csv", "Train dataset in csv file")
	testFile := flag.String("test", "test.csv", "Test dataset in csv file")
	outFile := flag.String("out", "out.csv", "Output file name")

	flag.Parse()

	if err := checkParams(numClasses, width, height, maxEpochs, learningRate); err != nil {
		fmt.Printf("%s", err)
		os.Exit(1)
	}

	p := perceptron.NewPerceptron(*numClasses, int32((*width)*(*height)))

	trainData, err := perceptron.ReadDataset(*trainFile, -1)
	if err != nil {
		fmt.Printf("Read dataset error: %s\n", err)
		return
	}
	trainData = perceptron.Normalize(trainData, 255)

	trainer := perceptron.Trainer{Perceptron: p, LearningRate: *learningRate}
	trainer.Train(trainData, *maxEpochs)

	testData, err := perceptron.ReadDataset(*testFile, -1)

	result := trainer.Perceptron.Recognize(testData)
	perceptron.WriteResult(*outFile, result)
}
