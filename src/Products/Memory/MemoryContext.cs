#pragma warning disable SKEXP0001, SKEXP0003, SKEXP0010, SKEXP0011, SKEXP0050, SKEXP0052, SKEXP0070, KMEXP01

using DataEntities;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.VectorData;
using Microsoft.KernelMemory;
using Microsoft.SemanticKernel.Connectors.InMemory;
using Newtonsoft.Json;
using OpenAI.Chat;
using OpenAI.Embeddings;
using Products.Models;
using SearchEntities;
using System.Text;
using System.Text.RegularExpressions;
using VectorEntities;

namespace Products.Memory;

public class MemoryContext(
    ILogger _logger,
    ChatClient? chatClientOpenAI,
    ChatClient? chatClientReasoningModel,
    EmbeddingClient? _embeddingClient,
    IKernelMemory kernelMemory)
{
    private readonly string _systemPrompt = "You are a useful assistant. You always reply with a short and funny message. If you do not know an answer, you say 'I don't know that.' You only answer questions related to outdoor camping products. For any other type of questions, explain to the user that you only answer outdoor camping products questions. Do not store memory of the chat conversation.";
    private bool _isMemoryCollectionInitialized = false;

    private IVectorStoreRecordCollection<int, ProductVector>? _productsCollection;

    public async Task<bool> InitMemoryContextAsync(Context db)
    {
        if (_isMemoryCollectionInitialized)
        {
            _logger.LogInformation("Memory context already initialized");
            return true;
        }

        _logger.LogInformation("Initializing memory context");
        var vectorProductStore = new InMemoryVectorStore();
        _productsCollection = vectorProductStore.GetCollection<int, ProductVector>("products");
        await _productsCollection.CreateCollectionIfNotExistsAsync();

        _logger.LogInformation("Get a copy of the list of products");
        var products = await db.Product.ToListAsync();

        _logger.LogInformation("Filling products in memory");

        // iterate over the products and add them to the memory
        foreach (var product in products)
        {
            try
            {
                _logger.LogInformation("Adding product to memory: {Product}", product.Name);
                var productInfo = $"[{product.Name}] is a product that costs [{product.Price}] and is described as [{product.Description}]";

                // new product vector
                var productVector = new ProductVector
                {
                    Id = product.Id,
                    Name = product.Name,
                    Description = product.Description,
                    Price = product.Price,
                    ImageUrl = product.ImageUrl
                };
                var result = await _embeddingClient!.GenerateEmbeddingAsync(productInfo);

                productVector.Vector = result.Value.ToFloats();
                var recordId = await _productsCollection.UpsertAsync(productVector);
                _logger.LogInformation("Product added to memory: {Product} with recordId: {RecordId}", product.Name, recordId);
            }
            catch (Exception exc)
            {
                _logger.LogError(exc, $"Error adding product {product.Name} to memory");
                _isMemoryCollectionInitialized = false;
                return false;
            }
        }

        _isMemoryCollectionInitialized = true;
        _logger.LogInformation("DONE! Filling products in memory");
        return true;
    }

    public async Task<SearchResponse> Search(string search, Context db, bool useReasoningModel = false)
    {
        if (!_isMemoryCollectionInitialized)
        {
            await InitMemoryContextAsync(db);
        }

        var response = new SearchResponse
        {
            Response = $"I don't know the answer for your question. Your question is: [{search}]"
        };

        try
        {
            var result = await _embeddingClient!.GenerateEmbeddingAsync(search);
            var vectorSearchQuery = result.Value.ToFloats();

            var searchOptions = new VectorSearchOptions<ProductVector>
            {
                Top = 3
            };

            // search the vector database for the most similar product        
            var searchResults = await _productsCollection!.VectorizedSearchAsync(vectorSearchQuery, searchOptions);
            var sbFoundProducts = new StringBuilder();
            int productPosition = 1;
            await foreach (var searchItem in searchResults.Results)
            {
                if (searchItem.Score > 0.5)
                {
                    var product = await db.FindAsync<Product>(searchItem.Record.Id);
                    if (product != null)
                    {
                        response.Products.Add(product);
                        sbFoundProducts.AppendLine($"- Product {productPosition}:");
                        sbFoundProducts.AppendLine($"  - Name: {product.Name}");
                        sbFoundProducts.AppendLine($"  - Description: {product.Description}");
                        sbFoundProducts.AppendLine($"  - Price: {product.Price}");
                        productPosition++;
                    }
                }
            }

            //        // get the 1st relevant product
            //        var answerProductId = int.TryParse(firstSource?.DocumentId, out var id) ? id : 0;
            //        if (answerProductId > 0)
            //        {
            //            var firstProduct = await db.Product.FindAsync(answerProductId);
            //            response.Products = new List<Product> { firstProduct };
            //            response.Response = $"The product [{firstProduct.Name}] fits with the search criteria [{search}]";
            //            _logger.LogInformation($"Search Response: {response.Response}");
            //        }

            //        // get the 2nd element from answer.RelevantSources.
            //        var secondSource = answer.RelevantSources.ElementAtOrDefault(1);
            //        if (secondSource != null)
            //        {
            //            var secondProductId = int.TryParse(secondSource.DocumentId, out var secondProdid) ? secondProdid : 0;
            //            if (secondProductId > 0)
            //            {
            //                var secondProduct = await db.Product.FindAsync(secondProductId);
            //                response.Products.Add(secondProduct);
            //                promptSecondProductInfo = @$"
            //- Found Second Product Name: {secondProduct?.Name}
            //- Found Second Product Description: {secondProduct?.Description}
            //- Found Second Product Price: {secondProduct?.Price}";
            //            }
            //        }

            // let's improve the response message
            var prompt = @$"You are an intelligent assistant helping clients with their search about outdoor products. 
Generate a catchy and friendly message using the information below.
Add a comparison between the products found and the search criteria.
Include products details.
    - User Question: {search}
    - Found Products: 
{sbFoundProducts}";

            var messages = new List<ChatMessage>
            {
                new SystemChatMessage(_systemPrompt),
                new UserChatMessage(prompt)
            };

            _logger.LogInformation("{ChatHistory}", JsonConvert.SerializeObject(messages));

            if (!useReasoningModel)
            {
                _logger.LogInformation("Generate response using standard chat model");
                var resultPrompt = await chatClientOpenAI.CompleteChatAsync(messages);
                response.Response = resultPrompt.Value.Content[0].Text!;
            }
            else
            {
                _logger.LogInformation("Generate response using reasoning model");
                var resultPrompt = await chatClientReasoningModel.CompleteChatAsync(messages);
                var responseComplete = resultPrompt.Value.Content[0].Text!;

                var match = Regex.Match(responseComplete, @"<think>(.*?)<\/think>(.*)", RegexOptions.Singleline);
                if (match.Success)
                {
                    response.ResponseThink = match.Groups[1].Value.Trim();
                    response.Response = match.Groups[2].Value.Trim();
                }
            }
        }
        catch (Exception ex)
        {
            response.Response = $"An error occurred: {ex.Message}";
            _logger.LogError(ex, "Error during search");
        }

        return response;
    }
}
