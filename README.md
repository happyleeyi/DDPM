# DDPM model

논문명 : Denoising Diffusion Probabilistic Models

## Training
- U-net base model
- main.py에서 여러 parameter 조절 가능
- kaggle, huggingface에서 데이터 받아와 training

## Sampling
- dt만큼씩 sampling 진행 후 저장 (dt는 한번에 진행하는 sampling step 수이며, 이것이 너무 크면 gpu 메모리 초과 가능)
- 저장된 numpy 파일 불러서 이어서 sampling 하는 방식
- 가장 기초적인 DDPM 모델이기에 time steps 수대로 모두 sampling
  
