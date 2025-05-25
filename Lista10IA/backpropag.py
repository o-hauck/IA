import numpy as np

class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden_neurons, num_output_neurons, learning_rate, activation_function_name, use_bias=True):
        self.num_inputs = num_inputs
        self.num_hidden_neurons = num_hidden_neurons
        self.num_output_neurons = num_output_neurons
        self.learning_rate = learning_rate
        self.use_bias = use_bias

        # Inicializa pesos aleatoriamente
        # np.random.seed(42) # Descomente para reprodutibilidade consistente entre execuções
        self.weights_input_hidden = np.random.uniform(-0.5, 0.5, (self.num_inputs, self.num_hidden_neurons))
        self.weights_hidden_output = np.random.uniform(-0.5, 0.5, (self.num_hidden_neurons, self.num_output_neurons))

        if self.use_bias:
            self.bias_hidden = np.random.uniform(-0.5, 0.5, (1, self.num_hidden_neurons))
            self.bias_output = np.random.uniform(-0.5, 0.5, (1, self.num_output_neurons))
        else:
            self.bias_hidden = np.zeros((1, self.num_hidden_neurons))
            self.bias_output = np.zeros((1, self.num_output_neurons))

        self.activation_function, self.derivative_activation_function = self._get_activation_function(activation_function_name)

    def _get_activation_function(self, name):
        if name == 'sigmoid':
            return self._sigmoid, self._sigmoid_derivative
        elif name == 'tanh':
            return self._tanh, self._tanh_derivative
        elif name == 'relu':
            return self._relu, self._relu_derivative
        else:
            raise ValueError("Função de ativação não suportada. Escolha entre 'sigmoid', 'tanh', 'relu'.")

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _sigmoid_derivative(self, x_output): # x_output é a saída da sigmoide
        return x_output * (1 - x_output)

    def _tanh(self, x):
        return np.tanh(x)

    def _tanh_derivative(self, x_output): # x_output é a saída da tanh
        return 1 - (x_output ** 2)

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x_output): # x_output é a saída da relu
        return (x_output > 0).astype(float)

    def feedforward(self, inputs):
        self.hidden_layer_input_net = np.dot(inputs, self.weights_input_hidden)
        if self.use_bias:
            self.hidden_layer_input_net += self.bias_hidden
        self.hidden_layer_output = self.activation_function(self.hidden_layer_input_net)

        self.output_layer_input_net = np.dot(self.hidden_layer_output, self.weights_hidden_output)
        if self.use_bias:
            self.output_layer_input_net += self.bias_output
        self.predicted_output = self.activation_function(self.output_layer_input_net)
        return self.predicted_output

    def backpropagate(self, inputs, targets):
        error_output = targets - self.predicted_output
        delta_output = error_output * self.derivative_activation_function(self.predicted_output)

        error_hidden = np.dot(delta_output, self.weights_hidden_output.T)
        delta_hidden = error_hidden * self.derivative_activation_function(self.hidden_layer_output)

        self.weights_hidden_output += self.learning_rate * np.dot(self.hidden_layer_output.T, delta_output)
        if self.use_bias:
            self.bias_output += self.learning_rate * np.sum(delta_output, axis=0, keepdims=True)

        self.weights_input_hidden += self.learning_rate * np.dot(inputs.T, delta_hidden)
        if self.use_bias:
            self.bias_hidden += self.learning_rate * np.sum(delta_hidden, axis=0, keepdims=True)

    def train(self, training_data, epochs, print_error_interval=0):
        self.epoch_errors = []
        for epoch in range(epochs):
            total_error_epoch = 0
            # np.random.shuffle(training_data) # Opcional: embaralhar dados a cada época
            for inputs_list, targets_list in training_data:
                inputs = np.array([inputs_list])
                targets = np.array([targets_list])

                self.feedforward(inputs)
                self.backpropagate(inputs, targets)
                total_error_epoch += np.mean(0.5 * (targets - self.predicted_output)**2) # MSE para acompanhamento

            avg_epoch_error = total_error_epoch / len(training_data)
            self.epoch_errors.append(avg_epoch_error)
            if print_error_interval > 0 and (epoch + 1) % print_error_interval == 0:
                print(f"Época {epoch+1}/{epochs}, Erro Médio Quadrático: {avg_epoch_error:.6f}")


    def predict(self, inputs_list):
        inputs = np.array([inputs_list])
        return self.feedforward(inputs)

def generate_boolean_data(num_inputs, gate_type):
    data = []
    for i in range(2**num_inputs):
        binary_representation = bin(i)[2:].zfill(num_inputs)
        inputs = [int(bit) for bit in binary_representation]
        target = 0
        if gate_type == 'AND':
            target = 1 if all(inputs) else 0
        elif gate_type == 'OR':
            target = 1 if any(inputs) else 0
        elif gate_type == 'XOR':
            target = sum(inputs) % 2
        else:
            raise ValueError("Tipo de porta lógica inválido.")
        data.append((inputs, [target]))
    return data

def run_experiment(gate_type, num_inputs, learning_rate, num_hidden_neurons, activation_function_name, epochs=10000, use_bias=True, print_details=True):
    header = f"--- {gate_type} com {num_inputs} entradas ({'COM' if use_bias else 'SEM'} Bias), LR={learning_rate}, Hidden={num_hidden_neurons}, Ativação={activation_function_name}, Épocas={epochs} ---"
    print(f"\n{header}\n" + "="*len(header))

    training_data = generate_boolean_data(num_inputs, gate_type)
    # Para consistência nos pesos iniciais ao variar APENAS um parâmetro (como LR), fixe o seed ANTES de cada init de NN relevante
    # np.random.seed(42) # Exemplo
    nn = NeuralNetwork(num_inputs, num_hidden_neurons, 1, learning_rate, activation_function_name, use_bias=use_bias)
    nn.train(training_data, epochs, print_error_interval=epochs//5 if epochs >=50 else 0) # Imprime erro 5x

    if print_details:
        print("\nResultados do Teste Detalhados:")
    correct_predictions = 0
    total_predictions = 0
    for inputs_list, target_list in training_data:
        prediction_value = nn.predict(inputs_list)[0][0]
        predicted_class = 1 if prediction_value >= 0.5 else 0
        if print_details:
            print(f"Entrada: {inputs_list}, Esperado: {target_list[0]}, Previsto (bruto): {prediction_value:.4f} (Classe: {predicted_class})")
        if predicted_class == target_list[0]:
            correct_predictions += 1
        total_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Acurácia Final: {accuracy:.2f}%")
    print("="*len(header) + "\n")
    return accuracy, nn.epoch_errors


if __name__ == '__main__':
    print("INÍCIO DOS EXPERIMENTOS COM BACKPROPAGATION")

    # --- DEMONSTRAÇÃO GERAL: AND, OR, XOR com n entradas ---
    print("\n\n--- SEÇÃO 1: Demonstração Geral (AND, OR, XOR com n entradas) ---")
    run_experiment('AND', 2, 0.1, 4, 'sigmoid', epochs=20000)
    run_experiment('OR', 3, 0.1, 6, 'sigmoid', epochs=25000)
    run_experiment('XOR', 2, 0.1, 4, 'tanh', epochs=30000) # MLP pode resolver XOR
    run_experiment('AND', 4, 0.1, 8, 'sigmoid', epochs=30000) # Aumentando n
    # Para 10 entradas, pode ser necessário ajustar neurônios/épocas e pode ser lento
    # run_experiment('AND', 10, 0.1, 20, 'sigmoid', epochs=50000, print_details=False) # print_details=False para não poluir muito
    # run_experiment('OR', 10, 0.1, 20, 'sigmoid', epochs=50000, print_details=False)
    run_experiment('XOR', 3, 0.1, 8, 'tanh', epochs=50000) # XOR com mais entradas

    # --- INVESTIGANDO A IMPORTÂNCIA DA TAXA DE APRENDIZADO ---
    print("\n\n--- SEÇÃO 2: Investigando a Importância da Taxa de Aprendizado ---")
    print("(Usando AND com 2 entradas, 4 neurônios ocultos, sigmoide, 20000 épocas)")
    learning_rates_to_test = [0.001, 0.01, 0.1, 0.5, 1.0]
    for lr in learning_rates_to_test:
        np.random.seed(42) # Resetar seed para comparar LRs com os mesmos pesos iniciais
        run_experiment('AND', 2, lr, 4, 'sigmoid', epochs=20000, print_details=(lr==0.1)) # Detalhes só para uma taxa

    # --- INVESTIGANDO A IMPORTÂNCIA DO BIAS ---
    print("\n\n--- SEÇÃO 3: Investigando a Importância do Bias ---")
    print("(Usando XOR com 2 entradas, 4 neurônios ocultos, tanh, LR=0.1, 30000 épocas)")
    print("Nota: A importância do bias é crucial. Sem ele, a rede pode não conseguir aprender funções que não são separáveis pela origem.")
    np.random.seed(42)
    run_experiment('XOR', 2, 0.1, 4, 'tanh', epochs=30000, use_bias=True)
    np.random.seed(42) # Mesmos pesos iniciais para comparação justa
    run_experiment('XOR', 2, 0.1, 4, 'tanh', epochs=30000, use_bias=False)
    # Tente também com uma função mais simples que o bias ainda ajuda
    np.random.seed(42)
    run_experiment('OR', 2, 0.1, 2, 'sigmoid', epochs=10000, use_bias=True)
    np.random.seed(42)
    run_experiment('OR', 2, 0.1, 2, 'sigmoid', epochs=10000, use_bias=False)


    # --- INVESTIGANDO A IMPORTÂNCIA DA FUNÇÃO DE ATIVAÇÃO ---
    print("\n\n--- SEÇÃO 4: Investigando a Importância da Função de Ativação ---")
    print("(Usando XOR com 2 entradas, 4 neurônios ocultos, LR=0.1, 30000 épocas)")
    activation_functions_to_test = ['sigmoid', 'tanh', 'relu']
    # Ajuste a taxa de aprendizado para ReLU se necessário, pois pode ser sensível
    learning_rate_for_activations = {'sigmoid': 0.1, 'tanh': 0.1, 'relu': 0.05}
    for act_func in activation_functions_to_test:
        np.random.seed(42) # Resetar seed
        run_experiment('XOR', 2, learning_rate_for_activations[act_func], 4, act_func, epochs=30000, print_details=(act_func=='sigmoid'))

    print("\nFIM DOS EXPERIMENTOS")