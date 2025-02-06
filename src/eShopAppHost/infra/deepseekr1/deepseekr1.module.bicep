@description('The location for the resource(s) to be deployed.')
param location string = resourceGroup().location

param rgname string = resourceGroup().name

param principalId string

param principalType string

var tags = { 'azd-env-name': rgname }

var aiServicesNameAndSubdomain = '${rgname}-aiservices'
module aiServices 'br/public:avm/res/cognitive-services/account:0.9.2' = {
  name: 'DeepSeek-R1'
  scope: resourceGroup()
  params: {
    name: aiServicesNameAndSubdomain
    location: location
    tags: tags
    kind: 'AIServices'
    customSubDomainName: aiServicesNameAndSubdomain
    sku:  'S0'
    publicNetworkAccess: 'Enabled'
    deployments: [
      {
        name: 'DeepSeek-R1'
        model: {
          format: 'DeepSeek'
          name: 'DeepSeek-R1'
          version: '1'
        }
        sku: {
          name: 'GlobalStandard'
          capacity: 1
        }
      }]
    disableLocalAuth: false
    roleAssignments: [
      {
        principalId: principalId
        principalType: principalType
        roleDefinitionIdOrName: 'Cognitive Services User'
      }
    ]
  }
}

output connectionString string = 'Endpoint=${aiServices.outputs.endpoint}'
