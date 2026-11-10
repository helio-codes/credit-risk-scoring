# Modelagem Preditiva de Risco de Crédito

Este repositório contém uma solução ponta a ponta de aprendizado supervisionado para prever probabilidade de default (inadimplência) com base em variáveis cadastrais e comportamentais de clientes.

## 📌 Objetivos
- Tratar dados tabulares com alta taxa de valores nulos e assimetria.
- Balancear classes minoritárias com técnicas de oversampling (SMOTE).
- Otimizar modelos de gradiente boosting focando nas métricas **ROC-AUC** e **Recall**.

## 🛠️ Tecnologias Utilizadas
- **Linguagem:** Python 3.10+
- **Bibliotecas:** pandas, scikit-learn, imbalanced-learn, XGBoost, Seaborn

## 📊 Métricas de Desempenho
O classificador final atingiu:
- **ROC-AUC:** ~0.84
- **Recall da classe default:** 0.76 (priorizando mitigar falsos negativos para controle de perdas financeiras)

## 🚀 Como Executar

1. Clone o repositório:
```bash
git clone [https://github.com/seu-usuario/credit-risk-scoring.git](https://github.com/seu-usuario/credit-risk-scoring.git)
cd credit-risk-scoring