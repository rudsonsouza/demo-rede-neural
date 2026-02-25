import tf from '@tensorflow/tfjs-node';

async function trainModel(inputXs, outputYs) {
    // Criamos um modelo sequencial simples
    const model = tf.sequential();

    // Primeira camada da rede:
    // entrada de 7 posições (idade normalizada + 3 cores + 3 localizações) 

    // 80 neuronios = porque tem pouca base de treino
    // quanto mais neuronios, mais complexo o modelo,
    // e mais processamento ela vai usar.

    // ReLU é uma função de ativação que ajuda a rede a aprender padrões complexos.
    // Como se ela deixasse somente os valores positivos passarem, 
    // ajudando a rede a focar em características importantes.
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }));

    // Camada de saída: 3 neurônios (para as 3 categorias) e função de ativação softmax
    // Softmax é usada para classificação, pois transforma as saídas em probabilidades.
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
    
    // Compilamos o modelo com otimizador Adam e função de perda categoricalCrossentropy
    // Adam é um otimizador eficiente para muitos tipos de problemas.
    // Categorical Crossentropy é adequada para classificação multiclasse.
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    // Treinamento do modelo com os dados de entrada (inputXs) e saída (outputYs)
    // epochs: número de vezes que o modelo verá todo o conjunto de dados durante o treinamento.
    // shuffle: embaralha os dados a cada época para melhorar o aprendizado.
    await model.fit(inputXs, outputYs, {
        epochs: 100,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`);
            }
        }
    });

    return model;
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

const model = trainModel(inputXs, outputYs);