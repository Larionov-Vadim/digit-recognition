package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
)

type neuron struct {
	bias    float64
	weights []float64
}

func newNeuron(numInputs int32) neuron {
	n := neuron{}
	n.weights = make([]float64, numInputs)
	return n
}

func (n neuron) activate(inputs []float64) float64 {
	sum := n.bias
	for index, weight := range n.weights {
		sum += weight * inputs[index]
	}

	return sigmoid(sum)
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-1*x))
}

func (n neuron) updateWeights(inputs []float64, learningRate float64, delta float64) {
	lenWeights := len(n.weights)
	for i := 0; i < lenWeights; i++ {
		n.weights[i] += learningRate * delta * inputs[i]
	}
	n.bias += delta * learningRate
}

type perceptron struct {
	neurons []neuron
}

func newPerceptron(numClasses int, numInputs int32) perceptron {
	p := perceptron{}
	p.neurons = make([]neuron, numClasses)
	for i := 0; i < numClasses; i++ {
		p.neurons[i] = newNeuron(numInputs)
	}
	return p
}

func (p perceptron) activate(inputs []float64) []float64 {
	result := make([]float64, len(p.neurons))
	for i, n := range p.neurons {
		result[i] = n.activate(inputs)
	}

	return result
}

func (p perceptron) classNumber(inputs []float64) int {
	probabilities := p.activate(inputs)
	return indexMaxElement(probabilities)
}

func indexMaxElement(slice []float64) int {
	index := 0
	maxElement := slice[0]
	for i, element := range slice {
		if element > maxElement {
			index = i
		}
	}
	return index
}

func (p perceptron) trainDataset(inputs []float64, expectedClass int, learningRate float64) bool {
	probabilities := p.activate(inputs)
	right := false
	if actualClass := indexMaxElement(probabilities); actualClass == expectedClass {
		right = true
	}

	for index, actualProbability := range probabilities {
		expectedProbability := 0.0
		if index == expectedClass {
			expectedProbability = 1.0
		}
		delta := expectedProbability - actualProbability
		if delta != 0 {
			p.neurons[index].updateWeights(inputs, learningRate, delta)
		}
	}

	return right
}

func (p perceptron) train(data [][]float64, learningRate float64, maxEpochs int) {
	lenData := len(data)
	for epoch := 0; epoch < maxEpochs; epoch++ {
		successCount := 0
		for _, dataset := range data {
			success := p.trainDataset(dataset[1:], int(dataset[0]), learningRate)
			if success {
				successCount++
			}
		}

		rightRecognized := float64(successCount) / float64(lenData) * 100
		fmt.Printf("Right recognized %f\n", rightRecognized)
	}
}

func readDataset(fileName string, limit int) [][]float64 {
	f, err := os.Open(fileName)
	if err != nil {
		fmt.Printf("File %s not found\n", fileName)
		return make([][]float64, 0, 0)
	}

	r := csv.NewReader(bufio.NewReader(f))

	// Skip header
	if _, err := r.Read(); err == io.EOF {
		return make([][]float64, 0, 0)
	}

	unlimited := false
	if limit == -1 {
		unlimited = true
	}

	readedCounter := 0
	var data [][]float64
	for i := 0; i < limit || unlimited; i++ {
		record, err := r.Read()
		if err == io.EOF {
			break
		}

		dataset := make([]float64, len(record))
		for j, valueString := range record {
			value, _ := strconv.ParseFloat(valueString, 64)
			dataset[j] = value
		}

		data = append(data, dataset)
		readedCounter++
		if readedCounter%1000 == 0 {
			fmt.Printf("Readed %d records\n", readedCounter)
		}
	}
	return data
}

func normalize(data [][]float64, maxValue float64) [][]float64 {
	for _, dataset := range data {
		for i, value := range dataset {
			if i != 0 {
				dataset[i] = value / maxValue
			}
		}
	}
	return data
}

func main() {
	numClasses := 10
	numInputs := (int32)(28 * 28)
	learningRate := 0.001
	maxEpochs := 10
	p := newPerceptron(numClasses, numInputs)

	testData := readDataset("train.csv", 10000)
	testData = normalize(testData, 255)

	p.train(testData, learningRate, maxEpochs)
	fmt.Printf("%v", p)
}
