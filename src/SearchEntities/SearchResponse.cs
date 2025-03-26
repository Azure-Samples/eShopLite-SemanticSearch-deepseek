using System.Text.Json.Serialization;

namespace SearchEntities;

public class SearchResponse
{
    public SearchResponse()
    {
        Products = new List<DataEntities.Product>();
    }

    [JsonPropertyName("response")]
    public string? Response { get; set; }

    [JsonPropertyName("responseThink")]
    public string? ResponseThink { get; set; }

    [JsonPropertyName("responseComplete")]
    public string? ResponseComplete { get; set; }

    [JsonPropertyName("products")]
    public List<DataEntities.Product>? Products { get; set; }

    [JsonPropertyName("elapsedTime")]
    public TimeSpan ElapsedTime { get; set; }
}


[JsonSerializable(typeof(SearchResponse))]
public sealed partial class SearchResponseSerializerContext : JsonSerializerContext
{
}