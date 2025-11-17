!!! abstract "Informações da entrega"
    📆 Deadline: 19/11/2025

    📖 O enunciado da atividade está disponível neste [link](https://insper.github.io/ann-dl/versions/2025.2/projects/generative).

As pipelines de modelos utilizadas nesse projeto foram feitas com base em modelos previamente treinados utilizando a ferramenta visual ComfyUI[^1]

## Text-to-image

A pipeline utilizada nesse exemplo é encontrada na documentação[^2].

Para exemplificar o funcionamento desse workflow, utilizaremos como base o diagrama da [Figura 1](#figure-1).

<a id="figure-1" style="text-decoration: none; color: inherit; justify-content:center;">

| ![text-to-image-workflow](./img/text_to_image.drawio.svg) |
| :--: |
| **Figura 1**: Fluxograma do processo de geração de imagem a partir de um texto. **Fonte**: Autor. |

</a>

Como apontado anteriormente, o modelo foi pré-treinado e seus pesos carregados em um arquivo no formato `safetensors`, que é um tipo de arquivo utilizado como alternativa aos de formato `pickle`, visto que apresenta os valores numéricos em forma de código executável[^4]. Os parâmetros do modelo são carregados devidamente em cada um dos componentes da pipeline: no CLIP, no KSampler e no VAE.

Como podemos observar, o processo inicia com o condicionamento do modelo com *Contrastive Language–Image Pre-training* (CLIP)[^3], seja ele positivo, ou seja, inserindo características que **são** desejadas na imagem a ser gerada, ou negativo, inserindo características **não** desejadas na imagem de saída.

Juntamente à entrada, que é uma imagem com ruído, os condicionamentos servem de entrada para o KSampler, que aplica o [modelo treinado](https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly-fp16.safetensors?download=true) para remover o ruído da imagem no espaço latente. Dessa forma, a imagem sem ruídos é encaminhada para o decoder do *Variational Auto Encoder* (VAE), que decodifica a imagem no espaço latente para uma imagem no formato original. 

O funcionamento da ferramenta pode ser visualizado pelo [Vídeo 1](#video-1).

<a id="video-1" style="text-decoration: none; color: inherit; justify-content:center;">

| <iframe width="560" height="315" src="https://www.youtube.com/embed/8g8q5ul2rbw?si=AZb5PfP9nk11oGCn" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe> |
| :--: |
| **Vídeo 1**: Funcionamento da ferramenta de acordo com as especificações. **Fonte**: Autor. |

</a>

[^1]:
    [ComfyUI | Generate video, images, 3D, audio with AI](https://www.comfy.org)

[^2]:
    [ComfyUI Text to Image Workflow](https://docs.comfy.org/tutorials/basic/text-to-image)

[^3]:
    [CLIP: Connecting text and images](https://openai.com/index/clip)

[^4]:
    [Safetensors](https://huggingface.co/docs/safetensors/index)