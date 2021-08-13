package deep

import (
	"fmt"
	"math/rand"
	"time"

	math "github.com/chewxy/math32"
	model "github.com/nathanleary/neural-net-test10/training/model"
)

// Example is an input-target pair
type Example struct {
	Input    []float32
	Response []float32
}

// Examples is a set of input-output pairs
type Examples []Example

// Shuffle shuffles slice in-place
func (e Examples) Shuffle() {
	for i := range e {
		j := rand.Intn(i + 1)
		e[i], e[j] = e[j], e[i]
	}
}

// Split assigns each element to two new slices
// according to probability p
func (e Examples) Split(p float32) (first, second Examples) {
	for i := 0; i < len(e); i++ {
		if p > rand.Float32() {
			first = append(first, e[i])
		} else {
			second = append(second, e[i])
		}
	}
	return
}

// SplitSize splits slice into parts of size size
func (e Examples) SplitSize(size int) []Examples {
	res := make([]Examples, 0)
	for i := 0; i < len(e); i += size {
		res = append(res, e[i:min(i+size, len(e))])
	}
	return res
}

// SplitN splits slice into n parts
func (e Examples) SplitN(n int) []Examples {
	res := make([]Examples, n)
	for i, el := range e {
		res[i%n] = append(res[i%n], el)
	}
	return res
}

func min(a, b int) int {
	if a <= b {
		return a
	}
	return b
}

// Neural is a neural network
type Neural struct {
	Layers []*Layer
	Biases [][]*Synapse
	Config *Config
}

// Config defines the network topology, activations, losses etc
type Config struct {
	// Number of inputs
	Inputs int
	// Defines topology:
	// For instance, [5 3 3] signifies a network with two hidden layers
	// containing 5 and 3 nodes respectively, followed an output layer
	// containing 3 nodes.
	Layout []int
	// Activation functions: {ActivationTanh, ActivationReLU, ActivationSigmoid}
	Activation ActivationType
	// Solver modes: {ModeRegression, ModeBinary, ModeMultiClass, ModeMultiLabel}
	Mode Mode
	// Initializer for weights: {NewNormal(σ, μ), NewUniform(σ, μ)}
	Weight WeightInitializer `json:"-"`
	// Loss functions: {LossCrossEntropy, LossBinaryCrossEntropy, LossMeanSquared}
	Loss LossType
	// Apply bias nodes
	Bias bool
	//Update Bonds
	Bonds float32
}

func (n *Neural) Refine(data model.Examples, lr float32) float32 {

	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	training := data //.Split(1.0 / float32(len(data)) * 1.0)
	dist := n.calcDistance(training)

	for x := 0; x < 2; x++ {
		l := n.Layers[r.Intn(len(n.Layers))]
		// for _, l := range n.Layers {

		rj := r.Intn(len(l.Neurons))
		rk := r.Intn(len(l.Neurons[rj].In))

		//update weights
		random := (r.Float32()*2 - 1.0) * lr
		randInc := l.Neurons[rj].In[rk].Weight * random
		l.Neurons[rj].In[rk].Weight += randInc

		d := n.calcDistance(training)
		if dist >= d {
			dist = d
		} else {
			l.Neurons[rj].In[rk].Weight -= randInc * 2
			if dist >= d {
				dist = d
			} else {
				l.Neurons[rj].In[rk].Weight += randInc
			}
		}

		// update bonds

		rkb := r.Intn(len(l.Neurons[rj].In))

		for len(l.Neurons[rj].In[rk].Bond) <= rkb {
			l.Neurons[rj].In[rk].Bond = append(l.Neurons[rj].In[rk].Bond, 1.0)
		}

		random = r.Float32()*2 - 1.0
		randInc = l.Neurons[rj].In[rk].Bond[rkb] * random

		l.Neurons[rj].In[rk].Bond[rkb] += randInc

		d = n.calcDistance(training)
		if dist >= d {
			dist = d
		} else {
			l.Neurons[rj].In[rk].Bond[rkb] -= randInc * 2
			if dist >= d {
				dist = d
			} else {
				l.Neurons[rj].In[rk].Bond[rkb] += randInc
			}
		}
		// }
	}
	return dist
}

func (n *Neural) RefineWeights(data model.Examples, lr float32) float32 {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	training := data //data.Split(1.0 / float32(len(data)) * 1.0)
	dist := n.calcDistance(training)

	for x := 0; x < 2; x++ {
		l := n.Layers[r.Intn(len(n.Layers))]
		// for _, l := range n.Layers {

		rj := r.Intn(len(l.Neurons))
		rk := r.Intn(len(l.Neurons[rj].In))

		//update weights
		random := (r.Float32()*2 - 1.0) * lr
		randInc := l.Neurons[rj].In[rk].Weight * random
		l.Neurons[rj].In[rk].Weight += randInc

		d := n.calcDistance(training)
		if dist >= d {
			dist = d
		} else {
			l.Neurons[rj].In[rk].Weight -= randInc * 2
			if dist >= d {
				dist = d
			} else {
				l.Neurons[rj].In[rk].Weight += randInc
			}
		}

		// }
	}
	return dist
}

func (n *Neural) CalcDistance(training2 model.Examples) float32 {
	return n.calcDistance(training2)
}

func (n *Neural) RefineBonds(data model.Examples, lr float32) float32 {
	// fmt.Println("test")
	// time.Sleep(time.Millisecond)
	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	training := data //data.Split(1.0 / float32(len(data)) * 1.0)
	dist := n.calcDistance(training)

	for x := 0; x < 2; x++ {
		l := n.Layers[r.Intn(len(n.Layers))]
		// for _, l := range n.Layers {

		rj := r.Intn(len(l.Neurons))
		rk := r.Intn(len(l.Neurons[rj].In))

		// //update weights
		// random := (r.Float32()*2 - 1.0) * lr
		// randInc := l.Neurons[rj].In[rk].Weight * random
		// l.Neurons[rj].In[rk].Weight += randInc

		// d := n.calcDistance(training)
		// if dist >= d {
		// 	dist = d
		// } else {
		// 	l.Neurons[rj].In[rk].Weight -= randInc * 2
		// 	if dist >= d {
		// 		dist = d
		// 	} else {
		// 		l.Neurons[rj].In[rk].Weight += randInc
		// 	}
		// }

		// update bonds

		rkb := r.Intn(len(l.Neurons[rj].In))

		for len(l.Neurons[rj].In[rk].Bond) <= rkb {
			l.Neurons[rj].In[rk].Bond = append(l.Neurons[rj].In[rk].Bond, 1.0)
		}

		random := (r.Float32()*2 - 1.0) * lr
		randInc := random //l.Neurons[rj].In[rk].Bond[rkb] * random

		l.Neurons[rj].In[rk].Bond[rkb] += randInc

		d := n.calcDistance(training)
		if dist >= d {
			dist = d
		} else {
			l.Neurons[rj].In[rk].Bond[rkb] -= randInc * 2
			if dist >= d {
				dist = d
			} else {
				l.Neurons[rj].In[rk].Bond[rkb] += randInc
			}
		}
		// }
	}
	return dist
}

func (n *Neural) calcDistance(training2 model.Examples) float32 {
	var diff float32 = 0.0

	for _, v := range training2 {
		prediction := n.Predict(v.Input)
		for ri, _ := range v.Response {
			// if v.Response[ri] == 0.0 {
			// p := prediction[ri]
			// if p != 0.0 {
			// diff += math.Abs(p)
			// }
			// } else {
			diff += (math.Abs((prediction[ri]) - (v.Response[ri])))
			// }

			// fmt.Println(prediction[ri], v.Response[ri])
			// time.Sleep(time.Microsecond)

		}
	}

	return diff
}

// NewNeural returns a new neural network
func NewNeural(c *Config) *Neural {

	if c.Weight == nil {
		c.Weight = NewUniform(0.5, 0)
	}
	if c.Activation == ActivationNone {
		c.Activation = ActivationSigmoid
	}
	if c.Loss == LossNone {
		switch c.Mode {
		case ModeMultiClass, ModeMultiLabel:
			c.Loss = LossCrossEntropy
		case ModeBinary:
			c.Loss = LossBinaryCrossEntropy
		default:
			c.Loss = LossMeanSquared
		}
	}

	layers := initializeLayers(c)

	var biases [][]*Synapse
	if c.Bias {
		biases = make([][]*Synapse, len(layers))
		for i := 0; i < len(layers); i++ {
			if c.Mode == ModeRegression && i == len(layers)-1 {
				continue
			}
			biases[i] = layers[i].ApplyBias(c.Weight)
		}
	}

	return &Neural{
		Layers: layers,
		Biases: biases,
		Config: c,
	}
}

func initializeLayers(c *Config) []*Layer {
	layers := make([]*Layer, len(c.Layout))
	for i := range layers {
		act := c.Activation
		if i == (len(layers)-1) && c.Mode != ModeDefault {
			act = OutputActivation(c.Mode)
		}
		layers[i] = NewLayer(c.Layout[i], act)
	}

	for i := 0; i < len(layers)-1; i++ {
		layers[i].Connect(layers[i+1], c.Weight)
	}

	for _, neuron := range layers[0].Neurons {
		neuron.In = make([]*Synapse, c.Inputs)
		for i := range neuron.In {
			neuron.In[i] = NewSynapse(c.Weight())
		}
	}

	return layers
}

func (n *Neural) fire(training bool) {
	for _, b := range n.Biases {
		for _, s := range b {
			s.fire(1)
		}
	}
	for _, l := range n.Layers {
		l.fire(training)
	}
}

// Forward computes a forward pass
func (n *Neural) Forward(input []float32, training bool) error {
	if len(input) != n.Config.Inputs {
		return fmt.Errorf("Invalid input dimension - expected: %d got: %d", n.Config.Inputs, len(input))
	}
	for _, n := range n.Layers[0].Neurons {
		for i := 0; i < len(input); i++ {
			n.In[i].fire(input[i])
		}
	}
	n.fire(training)
	return nil
}

// Predict computes a forward pass and returns a prediction
func (n *Neural) Predict(input []float32) []float32 {

	n.Forward(input, false)

	outLayer := n.Layers[len(n.Layers)-1]
	out := make([]float32, len(outLayer.Neurons))
	for i, neuron := range outLayer.Neurons {
		out[i] = neuron.Value
	}
	return out
}

// NumWeights returns the number of weights in the network
func (n *Neural) NumWeights() (num int) {
	for _, l := range n.Layers {
		for _, n := range l.Neurons {
			num += len(n.In)
		}
	}
	return
}

func (n *Neural) String() string {
	var s string
	for _, l := range n.Layers {
		s = fmt.Sprintf("%s\n%s", s, l)
	}
	return s
}
