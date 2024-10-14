using Microsoft.ML.Transforms.Text;

// Initialize input data
var inputData = new [] {"What is AI?", "What is ML?"};

// Specify pretrained model
var embeddingGenerator = new MLNETEmbeddingGenerator(WordEmbeddingEstimator.PretrainedModelKind.GloVe50D);

// Generate embeddings
var embeddings = 
    await embeddingGenerator.GenerateAsync(inputData);

// Display embeddings
var i = inputData.Zip(embeddings, (i, e) => (i, e));

foreach(var (input, embedding) in i)
{
    Console.WriteLine($"{input} => {string.Join(", ", embedding.Vector.ToArray())}");
}