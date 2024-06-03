# Estudi computacional de la transició entre els estats de son i vigília

En aquest Treball de Final de Grau titulat “Estudi computacional de la transició entre els estats de son i vigília”, s’analitza com canvien aquests estats centrant-se en la dinàmica de les neurones piramidals. 
El treball utilitza aquest model per simular i estudiar aquests canvis complexos.

Es comparen els estats de son NREM i de vigília, amb els estats transitoris entre aquests, per veure com és la dinàmica de la transició. 
L'estudi es centra en la freqüència de dispar de les neurones piramidals, observant com aquestes canvien entre els diferents estats de consciència.

## Origen del Codi

Aquest projecte es basa en el codi original de [Farhad Razi (2024). Heterogeneous Synaptic Homeostasis](https://github.com/fraziphy/heterogeneous_synaptic_homeostasis?tab=readme-ov-file). 

He fet diverses modificacions per als meus estudis, incloent:

- Afegit de la funció `calculate_betas` per calcular les betas:
    ```python
    def calculate_betas(time):
        max_value = 1
        min_value = 2

        # Calculem el rang de temps on volem aplicar el "step"
        step_start = 1  # Segons
        step_end = 9    # Segons

        # Càlcul del temps en ms
        step_start_ms = step_start * 1000 / dt 
        step_end_ms = step_end * 1000 / dt

        # Càlcul de beta_ampa_intra
        growth_rate = (max_value - min_value) / ((step_end - step_start) * 1000 / dt)

        if time < step_start_ms:
            return min_value
        elif step_start_ms <= time <= step_end_ms:
            beta_ampa_intra = min_value + growth_rate * (time - step_start_ms)
            return beta_ampa_intra
        else:
            return max_value

- Modificació de la funció `ONE_TRIAL_INTEGRATION` per equilibrar `beta_gaba`.

    if beta_ampa_intra > 1:
            fixedpoint = np.load("../data/V_WAKE.npy")
            beta_gaba = FIND_BETA_GABA(beta_ampa_intra, beta_ampa_inter, fixedpoint)
  

## Instal·lació

Per utilitzar aquest projecte, segueix els següents passos:

1. Clona aquest repositori:
    ```bash
    git clone https://github.com/el-teu-usuari/nom-del-repositori.git
    ```

2. Navega al directori del projecte:
    ```bash
    cd nom-del-repositori
    ```

3. Instal·la les dependències requerides (si escau):
    ```bash
    pip install -r requirements.txt
    ```
