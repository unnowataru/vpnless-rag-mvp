# Terraform (prod)

## 初回セットアップ
```bash
cd infra/live/prod
cp terraform.tfvars.example terraform.tfvars
terraform init
terraform plan
terraform apply
```

## 既存 `infra/root` からの移行（ローカル state）
既に `infra/root/terraform.tfstate` を使っていた場合は、次を 1 回実施してください。

```bash
cd /home/user/dev/vpnless-rag-mvp
mkdir -p infra/live/prod
cp infra/root/terraform.tfstate* infra/live/prod/ 2>/dev/null || true
```

その後、`infra/live/prod` で `terraform init -reconfigure` を実行します。
