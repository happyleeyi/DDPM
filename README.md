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


## Result
- cat face에 대해 학습 진행
- 
![fig_25](https://github.com/user-attachments/assets/33c3da1a-8d6d-47dd-962c-b630f7138d75)
![fig_300](https://github.com/user-attachments/assets/a0057a62-6a5a-4e0b-ab9e-2d70a033f3f5)
![fig_700](https://github.com/user-attachments/assets/35765c17-5180-4a32-8463-41c2e209423e)
![fig_825](https://github.com/user-attachments/assets/52c2d34d-830f-4a41-ad5d-05406d6333b3)
![fig_1000](https://github.com/user-attachments/assets/e00a0b8c-71af-4211-9d78-8f880d9227aa)
