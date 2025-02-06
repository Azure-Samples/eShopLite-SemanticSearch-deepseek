#pragma warning disable SKEXP0001, SKEXP0003, SKEXP0010, SKEXP0011, SKEXP0050, SKEXP0052, SKEXP0070, KMEXP01

using Microsoft.EntityFrameworkCore;
using SearchEntities;
using DataEntities;
using OpenAI.Chat;
using Microsoft.SemanticKernel.Connectors.InMemory;
using Newtonsoft.Json;
using Products.Models;
using System.Text.RegularExpressions;
using Microsoft.KernelMemory;
using OpenAI.Embeddings;

namespace Products.Memory;

public class MemoryContext(
    ILogger logger,
    ChatClient? chatClientOpenAI,
    ChatClient? chatClientReasoningModel,
    EmbeddingClient? embeddingClient,
    IKernelMemory kernelMemory)
{
    private readonly string _systemPrompt = "You are a useful assistant. You always reply with a short and funny message. If you do not know an answer, you say 'I don't know that.' You only answer questions related to outdoor camping products. For any other type of questions, explain to the user that you only answer outdoor camping products questions. Do not store memory of the chat conversation.";
    private bool _isMemoryCollectionInitialized = false;

    public async Task<bool> InitMemoryContextAsync(Context db)
    {
        logger.LogInformation("Initializing memory context");

        var products = await db.Product.ToListAsync();
        logger.LogInformation("Filling products in memory");

        foreach (var product in products)
        {
            try
            {
                logger.LogInformation("Adding product to memory: {Product}", product.Name);
                var productInfo = $"[{product.Name}] is a product that costs [{product.Price}] and is described as [{product.Description}]";
                await kernelMemory.ImportTextAsync(productInfo, product.Id.ToString());
                logger.LogInformation("Product added to memory: {Product}", product.Name);
            }
            catch (Exception exc)
            {
                logger.LogError(exc, "Error adding product to memory");
            }
        }

        logger.LogInformation("DONE! Filling products in memory");
        return true;
    }

    public async Task<SearchResponse> Search(string search, Context db, bool useReasoningModel = false)
    {
        if (!_isMemoryCollectionInitialized)
        {
            await InitMemoryContextAsync(db);
            _isMemoryCollectionInitialized = true;
        }

        var response = new SearchResponse
        {
            Response = $"I don't know the answer for your question. Your question is: [{search}]"
        };

        try
        {
            var answer = await kernelMemory.AskAsync(search);
            var firstSource = answer.RelevantSources.FirstOrDefault();
            var promptSecondProductInfo = string.Empty;

            // get the 1st relevant product
            var answerProductId = int.TryParse(firstSource?.DocumentId, out var id) ? id : 0;
            if (answerProductId > 0)
            {
                var firstProduct = await db.Product.FindAsync(answerProductId);
                response.Products = new List<Product> { firstProduct };
                response.Response = $"The product [{firstProduct.Name}] fits with the search criteria [{search}]";
                logger.LogInformation($"Search Response: {response.Response}");
            }

            // get the 2nd element from answer.RelevantSources.
            var secondSource = answer.RelevantSources.ElementAtOrDefault(1);
            if (secondSource != null)
            {
                var secondProductId = int.TryParse(secondSource.DocumentId, out var secondProdid) ? secondProdid : 0;
                if (secondProductId > 0)
                {
                    var secondProduct = await db.Product.FindAsync(secondProductId);
                    response.Products.Add(secondProduct);
                    promptSecondProductInfo = @$"
    - Found Second Product Name: {secondProduct?.Name}
    - Found Second Product Description: {secondProduct?.Description}
    - Found Second Product Price: {secondProduct?.Price}";
                }
            }

            var prompt = @$"You are an intelligent assistant helping clients with their search about outdoor products. Generate a catchy and friendly message using the following information:
    - User Question: {search}
    - Found Product Name: {response.Products?.FirstOrDefault()?.Name}
    - Found Product Description: {response.Products?.FirstOrDefault()?.Description}
    - Found Product Price: {response.Products?.FirstOrDefault()?.Price}
    {promptSecondProductInfo}
Include the found product information in the response to the user question.
If there is a second product information, also include it in the response with a comparison between the two products.
Generate only text, do not generate markdown or html content.";

            var messages = new List<ChatMessage>
            {
                new SystemChatMessage(_systemPrompt),
                new UserChatMessage(prompt)
            };

            logger.LogInformation("{ChatHistory}", JsonConvert.SerializeObject(messages));

            if (!useReasoningModel)
            {
                logger.LogInformation("Generate response using standard chat model");
                var resultPrompt = await chatClientOpenAI.CompleteChatAsync(messages);
                response.Response = resultPrompt.Value.Content[0].Text!;
            }
            else
            {
                logger.LogInformation("Generate response using reasoning model");
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
        }

        return response;
    }
}
