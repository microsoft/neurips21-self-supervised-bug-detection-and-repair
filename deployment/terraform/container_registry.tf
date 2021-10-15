resource "azurerm_container_registry" "buglabcr" {
    name                = "${var.prefix}cr"
    resource_group_name = azurerm_resource_group.buglab.name
    location            = azurerm_resource_group.buglab.location
    sku                 = "Premium"
    retention_policy {
     enabled = true
     days    = 30
   }
}
