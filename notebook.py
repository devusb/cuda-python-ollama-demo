import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _(mo):
    mo.md("""
    # Jacobs ML Demo
    """)
    return


@app.cell
def _():
    import inference

    return (inference,)


@app.cell
def _(inference, mo):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = inference.models.resnet50(
        weights=inference.models.ResNet50_Weights.DEFAULT
    ).to(device).eval()

    input_tensor = torch.randn(1, 3, 224, 224).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    weights = inference.models.ResNet50_Weights.DEFAULT
    categories = weights.meta["categories"]
    top5 = torch.topk(output[0], 5)

    results = "\n".join(
        f"- **{categories[idx]}**: {prob:.1%}"
        for prob, idx in zip(top5.values.softmax(0), top5.indices)
    )
    mo.output.replace(
        mo.md(f"""
    ## PyTorch Image Classification

    **Device:** `{device}`

    **Top predictions:**
    {results}
    """)
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
