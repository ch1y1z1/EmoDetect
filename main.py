import onnx_predict
import gradio as gr

imgs = ["pics/" + str(i) + ".jpeg" for i in range(1, 25)] + ["pics/" + str(i) + ".png" for i in range(1, 3)] + ["pics/" + str(i) + ".webp" for i in range(1, 2)]


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    gr.Interface(fn=onnx_predict.onnx_predict,
                 inputs=gr.Image(type="pil"),
                 outputs=gr.Label(num_top_classes=4),
                 examples=imgs)\
                .launch(server_name="0.0.0.0")
