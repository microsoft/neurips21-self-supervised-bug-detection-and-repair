variable "client_secret" {}

variable "storage_account_key" {}

variable "client_id" { }

variable "subscription_id" { }

variable "tennant_id" { }

variable "agent_count" {
    default = 1
}

variable "prefix" {
    default = "buglab"
}

variable resource_group_name {
    default = "buglab"
}

variable location { }

variable log_analytics_workspace_name { }

# refer https://azure.microsoft.com/global-infrastructure/services/?products=monitor for log analytics available regions
variable log_analytics_workspace_location {}

# refer https://azure.microsoft.com/pricing/details/monitor/ for log analytics pricing
variable log_analytics_workspace_sku {
    default = "PerGB2018"
}
