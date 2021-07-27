
import numpy as np



class CompetetiveLearning():

    def __init__(self, input_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        #initializing random weights
        self.weights = np.random.uniform(low=0.0, high=1.0, size=(output_nodes, input_nodes)).round(1)


    def train(self, train_x):
        print("----training for "+str(len(train_x))+" samples------")
        clustering = {"A": [], "B": [], "C": []}
        count = 1
        for i in train_x:
            print("Iteration "+str(count))

            x = i.reshape((6, 1)) # reshaping the ith input value so matrix multiplication can be applied
            result = np.matmul(self.weights, x) #multiplying wieghts with input nodes (w11X1 + w21X2 + ....)
            winning_unit = result.argmax() # index with maximum value will be the winning unit (only row with these weights will be updated)

            print("Output Values for Iteration "+str(count)+": ")
            print(result)
            print("Winning Unit: "+str(winning_unit+1))
            print("Adjusting the weight for only row "+str(winning_unit+1))

            self.adjust_weights(0.5, winning_unit, x)

            clustering[list(clustering.keys())[winning_unit]].append("R"+str(count))
            count+=1

        self.print_final_weights()

        print("\nFinal Cluster Results: ")
        print(clustering)
    def print_final_weights(self):

        print("\nFinal Weights for Output P: ")
        print(self.weights[0])
        print("Final Weights for Output Q: ")
        print(self.weights[1])
        print("Final Weights for Output R: ")
        print(self.weights[2])



    def adjust_weights(self, learning_rate, row_no, inputs):

        for i in range(len(self.weights[row_no])):
            #adjusting the weights
            self.weights[row_no][i] = self.weights[row_no][i] + learning_rate*inputs[i]
            #normalizing the weights
            self.weights[row_no][i]/=2

    def test(self, test_x):
        print()
        print("----testing for " + str(len(test_x)) + " samples------")
        print()
        count = 1
        classes = ["Class A", "Class B", "Class C"]
        for i in test_x:
            print("Iteration " + str(count))

            x = i.reshape((6, 1))  # reshaping the ith input value so matrix multiplication can be applied
            result = np.matmul(self.weights, x)  # multiplying wieghts with input nodes (w11X1 + w21X2 + ....)
            winning_unit = result.argmax()  # index with maximum value will be the winning unit (only row with these weights will be updated)

            print("Output Values for t" + str(count) + ": ")
            print(result)
            print("Winning Unit: " + str(winning_unit + 1))


            print("t"+str(count)+" belongs to "+classes[winning_unit])

            count += 1


cl = CompetetiveLearning(6, 3)
train_x = np.array([

    [1, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 0, 1],
    [0, 0, 1, 1, 0, 1],
    [0, 0, 1, 0, 0, 1],
    [0, 1, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 1],
    [0, 1, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 0]

])

test_x = np.array([

    [0, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 1],
    [0, 1, 0, 1, 0, 1]

])
cl.train(train_x)
cl.test(test_x)