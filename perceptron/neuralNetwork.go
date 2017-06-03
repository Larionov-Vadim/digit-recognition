package perceptron

import (
	"fmt"
	"math"
)

type Neuron struct {
	bias       float64
	weights    []float64
	activation func(sum float64) float64
}

func newNeuron(numInputs int32) *Neuron {
	n := Neuron{}
	n.weights = make([]float64, numInputs)
	n.activation = sigmoid
	return &n
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func (n *Neuron) activate(inputs []float64) float64 {
	sum := -n.bias
	for index := range n.weights {
		sum += n.weights[index] * inputs[index]
	}
	return n.activation(sum)
}

func (n *Neuron) updateWeights(inputs []float64, learningRate, delta float64) {
	deltaBias := delta * learningRate
	n.bias -= deltaBias
	for i := range n.weights {
		n.weights[i] += deltaBias * inputs[i]
	}
}

type Perceptron struct {
	neurons []*Neuron
}

func NewPerceptron(numClasses int, numInputs int32) *Perceptron {
	p := Perceptron{}
	p.neurons = make([]*Neuron, numClasses)
	for i := 0; i < numClasses; i++ {
		p.neurons[i] = newNeuron(numInputs)
	}
	return &p
}

func (p *Perceptron) activate(inputs []float64) []float64 {
	result := make([]float64, len(p.neurons))
	for i, n := range p.neurons {
		result[i] = n.activate(inputs)
	}

	return result
}

func (p *Perceptron) classNumber(inputs []float64) int {
	probabilities := p.activate(inputs)
	return indexMaxElement(probabilities)
}

func indexMaxElement(slice []float64) int {
	index := 0
	maxElement := slice[0]
	for i, element := range slice {
		if element > maxElement {
			maxElement = element
			index = i
		}
	}
	return index
}

func (p *Perceptron) updateWeights(inputs []float64, classNumber int, delta, learningRate float64) {
	p.neurons[classNumber].updateWeights(inputs, learningRate, delta)
}

type Trainer struct {
	Perceptron   *Perceptron
	LearningRate float64
}

func (t *Trainer) Train(trainData [][]float64, maxEpochs int) {
	for epoch := 0; epoch < maxEpochs; epoch++ {
		rightRecognized := t.trainEpoch(trainData, epoch)
		fmt.Printf("Epoch: %d; Right recognized: %f\n", epoch, float64(rightRecognized)/float64(len(trainData)))
	}
}

func (t *Trainer) trainEpoch(trainData [][]float64, epoch int) int {
	rightRecognized := 0
	for _, dataset := range trainData {
		activationResult := t.Perceptron.activate(dataset[1:])
		if int(dataset[0]) == indexMaxElement(activationResult) {
			rightRecognized++
		}

		for classNumber, r := range activationResult {
			var delta float64
			if classNumber != int(dataset[0]) {
				delta = 0.0 - r
			} else {
				delta = 1.0 - r
			}

			t.Perceptron.updateWeights(dataset[1:], classNumber, delta, t.LearningRate)
		}
	}
	return rightRecognized
}

func (p *Perceptron) Recognize(testData [][]float64) []int {
	result := make([]int, len(testData))
	for i, dataset := range testData {
		result[i] = p.classNumber(dataset)
	}
	return result
}
