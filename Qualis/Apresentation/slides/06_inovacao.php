<section class="slide" id="slide-6">

    <input type="checkbox" id="chk-simuladores" style="display: none;">

    <style>
        /* Estilos do Grid de Simuladores */
        .simulators-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-top: 20px; }
        .sim-card { background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 10px; overflow: hidden; transition: transform 0.3s, box-shadow 0.3s; }
        .sim-card:hover { transform: translateY(-5px); box-shadow: 0 10px 20px rgba(0, 0, 0, 0.5); border-color: rgba(148, 163, 184, 0.5); }
        .sim-img-container { width: 100%; height: 160px; background: #000; overflow: hidden; border-bottom: 3px solid; }
        .sim-img-container img { width: 100%; height: 100%; object-fit: cover; opacity: 0.8; transition: opacity 0.3s; }
        .sim-card:hover .sim-img-container img { opacity: 1; }
        .sim-content { padding: 20px; }
        .sim-content h4 { font-size: 1.2rem; margin-bottom: 10px; color: #f8fafc; }
        .sim-content p { font-size: 0.9rem; color: #cbd5e1; line-height: 1.4; }
        .border-sumo { border-color: #fbbf24; }
        .border-carla { border-color: #3b82f6; }
        .border-vissim { border-color: #f87171; }

        /* Lógica do Modal com Pure CSS (Sem depender de JavaScript) */
        #modal-simuladores-css {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(15, 23, 42, 0.95); backdrop-filter: blur(8px);
            z-index: 9999; display: flex; justify-content: center; align-items: center;
            opacity: 0; visibility: hidden; pointer-events: none; /* Escondido por padrão */
            transition: all 0.4s ease;
        }
        #modal-simuladores-css .cyber-modal-content {
            transform: translateY(20px); transition: transform 0.4s ease;
            background: #0f172a; border: 1px solid rgba(110, 231, 183, 0.3);
            border-radius: 12px; padding: 40px; width: 90%; max-width: 1100px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5); position: relative;
        }

        /* Quando o interruptor invisível é ativado, a mágica acontece */
        #chk-simuladores:checked ~ #modal-simuladores-css {
            opacity: 1; visibility: visible; pointer-events: auto;
        }
        #chk-simuladores:checked ~ #modal-simuladores-css .cyber-modal-content {
            transform: translateY(0);
        }
    </style>

    <div class="full-center fade-up">
        <h1 class="title-main">Inovação da Proposta</h1>
        <h2 class="subtitle">Sair de simuladores estáticos para um ecossistema dinâmico, preditivo e interativo.</h2>

        <div class="innovation-grid">
            <div class="innov-card green-hover">
                <div class="innov-icon"><i class="fas fa-satellite-dish"></i></div>
                <h4>1. Integração em Tempo Real</h4>
                <p>Dados ao vivo (tráfego, clima e status dos semáforos) alimentando a simulação sem defasagem de tempo.</p>
            </div>
            <div class="innov-card blue-hover">
                <div class="innov-icon"><i class="fas fa-microchip"></i></div>
                <h4>2. Sensores Virtuais Emulados</h4>
                <p>Sensores virtuais imitando componentes físicos reais (Câmeras, Triggers, Semáforos) com baixo custo e precisão.</p>
            </div>
            <div class="innov-card blue-hover">
                <div class="innov-icon"><i class="fas fa-brain"></i></div>
                <h4>3. Análise Preditiva (IA)</h4>
                <p>Utilização de algoritmos de aprendizado de máquina para prever eventos críticos, falhas e congestionamentos.</p>
            </div>
            <div class="innov-card green-hover">
                <div class="innov-icon"><i class="fas fa-project-diagram"></i></div>
                <h4>4. Interface Interativa</h4>
                <p>Ambiente visual e intuitivo que permite aos gestores visualizar impactos, testar cenários e tomar decisões rapidamente.</p>
            </div>
        </div>

        <div style="margin-top: 40px; display: flex; justify-content: center; gap: 20px;">
            <button id="btn-open-table" class="cyber-btn">
                <i class="fas fa-book-open" style="margin-right: 10px;"></i> Ver Comparativo da Literatura
            </button>

            <label for="chk-simuladores" class="cyber-btn" style="border-color: #fbbf24; color: #fbbf24; cursor: pointer; display: inline-flex; align-items: center;">
                <i class="fas fa-car-crash" style="margin-right: 10px;"></i> Simuladores Atuais
            </label>
        </div>

    </div>

    <div id="modal-compare" class="cyber-modal-overlay">
        <div class="cyber-modal-content">
            <button id="btn-close-table" class="modal-close"><i class="fas fa-times"></i></button>
            <h3 style="color: var(--neon-blue); font-size: 28px; margin-bottom: 20px; text-align: center;">Estado da Arte vs. SmartCitySystem</h3>

            <div class="table-responsive">
                <table class="cyber-table">
                    <thead>
                    <tr>
                        <th>Artigo</th>
                        <th>Tecnologias Usadas</th>
                        <th>O que se trata</th>
                        <th>Pontos Relevantes</th>
                    </tr>
                    </thead>
                    <tbody>
                    <tr>
                        <td><strong>BAO et al., 2021</strong></td>
                        <td>Aprendizado de máquina, Big Data, simulação</td>
                        <td>Revisão sistemática das aplicações de GD em transporte inteligente.</td>
                        <td>Discussão de desafios (integração de legados e qualidade de dados); foco na padronização.</td>
                    </tr>
                    <tr>
                        <td><strong>NIAZ et al., 2022</strong></td>
                        <td>Dispositivos IoT, sensores distribuídos</td>
                        <td>Modelo para monitoramento da infraestrutura e sincronização semafórica.</td>
                        <td>Ênfase em segurança rodoviária e economia de energia via sincronização dinâmica.</td>
                    </tr>
                    <tr>
                        <td><strong>ISODA et al., 2023</strong></td>
                        <td>Sensores embarcados, IA, simulador CARLA</td>
                        <td>Plataforma para gerenciar e otimizar fluxos de tráfego em tempo real.</td>
                        <td>Uso do CARLA para simulação dinâmica; foco em otimização de rotas e fluidez.</td>
                    </tr>
                    <tr>
                        <td><strong>GE et al., 2024</strong></td>
                        <td>Algoritmos de IA, sensores IoT, sistemas V2X</td>
                        <td>Modelo integrado com foco em veículos autônomos e infraestrutura.</td>
                        <td>Integração V2X (veículo para tudo); uso prático em veículos autônomos.</td>
                    </tr>
                    <tr class="highlight-row">
                        <td><i class="fas fa-star" style="color: var(--neon-green); margin-right: 5px;"></i> <strong>PROPOSTA (SmartCitySystem)</strong></td>
                        <td>Plataforma de Simulação (Unity), Sensores Virtuais, NoSQL</td>
                        <td>Plataforma de Digital Twins com sensores virtuais (semáforos, contadores) e APIs ao vivo (clima/hora).</td>
                        <td>Simulação integrada, baixo custo via sensores virtuais, arquitetura escalável IoT/Big Data.</td>
                    </tr>
                    </tbody>
                </table>
            </div>
            <p style="text-align: right; font-size: 12px; color: #64748b; margin-top: 10px;">Fonte: Elaborada pelo autor.</p>
        </div>
    </div>

    <div id="modal-simuladores-css">
        <div class="cyber-modal-content">

            <label for="chk-simuladores" style="position: absolute; top: 20px; right: 25px; color: #64748b; font-size: 28px; cursor: pointer;">
                <i class="fas fa-times"></i>
            </label>

            <h3 style="color: #fbbf24; font-size: 28px; margin-bottom: 10px; text-align: center;">O Cenário dos Simuladores</h3>
            <p style="text-align: center; color: #94a3b8; margin-bottom: 30px;">Análise crítica das ferramentas consolidadas no mercado.</p>

            <div class="simulators-grid">
                <div class="sim-card">
                    <div class="sim-img-container border-sumo">
                        <img src="https://user-images.githubusercontent.com/45896065/118984177-81859d00-b97d-11eb-99ca-5794c8672cb0.gif" alt="Eclipse SUMO">
                    </div>
                    <div class="sim-content">
                        <h4><i class="fas fa-map-marked-alt" style="color: #fbbf24; margin-right: 5px;"></i> Eclipse SUMO</h4>
                        <p>Considerado o padrão para simulação macroscópica. Possui interface 2D simples, mas apresenta uma curva de dificuldade altíssima para integração com IoT em tempo real.</p>
                    </div>
                </div>

                <div class="sim-card">
                    <div class="sim-img-container border-carla">
                        <img src="https://wuhanstudio.nyc3.cdn.digitaloceanspaces.com/blog/carla_tutorial/carla.gif" alt="CARLA Simulator">
                    </div>
                    <div class="sim-content">
                        <h4><i class="fas fa-car" style="color: #3b82f6; margin-right: 5px;"></i> CARLA</h4>
                        <p>Fotorrealismo impressionante focado no treinamento de IA para veículos autônomos. Ignora a infraestrutura macro da via e exige hardware de altíssimo custo para operar.</p>
                    </div>
                </div>

                <div class="sim-card">
                    <div class="sim-img-container border-vissim">
                        <img src="https://i.makeagif.com/media/1-31-2024/4CDR2i.gif" alt="PTV Vissim">
                    </div>
                    <div class="sim-content">
                        <h4><i class="fas fa-city" style="color: #f87171; margin-right: 5px;"></i> PTV VISSIM</h4>
                        <p>O padrão comercial da indústria para engenharia de tráfego. Extremamente detalhado, porém é um software proprietário, de código fechado e licenças de altíssimo custo.</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
</section>