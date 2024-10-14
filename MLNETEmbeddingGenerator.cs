using Microsoft.Extensions.AI;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;

// Reference implementation of ML.NET embedding generator
public class MLNETEmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
    private MLContext _mlContext;
    private ITransformer _pipeline;

    // Constructor that takes a custom IEstimator<ITransformer>
    public MLNETEmbeddingGenerator(MLContext mlContext, IEstimator<ITransformer> pipeline, IDataView dataView)
    {
        _mlContext = mlContext;
        _pipeline = pipeline.Fit(dataView);
    }

    // Constructor that takes a custom ITransformer
    public MLNETEmbeddingGenerator(MLContext mlContext, ITransformer pipeline)
    {
        _mlContext = mlContext;
        _pipeline = pipeline;
    }

    // Constructor that takes a save ML.NET pipeline from a file
    public MLNETEmbeddingGenerator(MLContext mlContext, string pipelinePath)
    {
        _mlContext = mlContext;
        _pipeline = mlContext.Model.Load(pipelinePath, out _);
    }

    // Constructor that takes a saved ML.NET pipeline from a stream
    public MLNETEmbeddingGenerator(MLContext mlContext, Stream modelStream)
    {
        _mlContext = mlContext;
        _pipeline = mlContext.Model.Load(modelStream, out _);
    }

    // Constructor that builds a pipeline based on the pretrained model specified
    public MLNETEmbeddingGenerator(WordEmbeddingEstimator.PretrainedModelKind modelKind)
    {
        _mlContext = new MLContext();
        _pipeline = 
            _mlContext.Transforms.Text.NormalizeText("NormalizedInputvalue","InputValue")
                .Append(_mlContext.Transforms.Text.TokenizeIntoWords("Tokens", "NormalizedInputvalue"))
                .Append(_mlContext.Transforms.Text.ApplyWordEmbedding("Embedding", "Tokens", modelKind))
                .Fit(_mlContext.Data.LoadFromEnumerable(new [] { new EmbeddingInput { InputValue = "Hello World" } }));
    }

    public EmbeddingGeneratorMetadata Metadata => new EmbeddingGeneratorMetadata(providerName: "MLNET Embedding Generator");

    public void Dispose()
    {
        throw new NotImplementedException();
    }

    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(IEnumerable<string> values, EmbeddingGenerationOptions? options = null, CancellationToken cancellationToken = default)
    {
        // Map string inputs to an IDataView
        var dataView = _mlContext.Data.LoadFromEnumerable(
                values.Select(x => new EmbeddingInput { InputValue = x }));
        
        // Apply pipeline to data
        var transformedData = _pipeline.Transform(dataView);
        
        // Extract embedding column and map it to Embedding<float>
        // var embeddings = 
        //     _mlContext.Data.CreateEnumerable<EmbeddingOutput>(transformedData, reuseRowObject: false)
        //         .Select(x => new Embedding<float>(x.Embedding));
        var embeddings = transformedData.ToGeneratedEmbeddings<float>("Embedding");

        // Return embeddings
        return Task.FromResult(embeddings);
    }

    public ITransformer? GetService<ITransformer>(object? key = null) where ITransformer : class
    {
        return _pipeline as ITransformer;
    }
}

public static class IDataViewEmbeddingExtensions
{
    public static GeneratedEmbeddings<Embedding<T>> ToGeneratedEmbeddings<T>(this IDataView dv, string columnName)
    {
        var embeddings = dv.GetColumn<float[]>(columnName);
        return new GeneratedEmbeddings<Embedding<T>>(
            embeddings.Select(x => new Embedding<T>(x.Cast<T>().ToArray())));
    }
}