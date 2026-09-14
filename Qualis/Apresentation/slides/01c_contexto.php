<section class="slide" id="slide-contexto">
    <style>
        /* Efeito de clique no núcleo do Digital Twin */
        .dt-core-interactive {
            cursor: pointer;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .dt-core-interactive:hover {
            transform: scale(1.1);
            box-shadow: 0 0 30px rgba(16, 185, 129, 0.6);
        }
        .dt-core-interactive::after {
            content: "Clique para expandir";
            position: absolute;
            bottom: -30px;
            font-size: 12px;
            color: #10b981;
            opacity: 0;
            transition: opacity 0.3s;
            white-space: nowrap;
        }
        .dt-core-interactive:hover::after {
            opacity: 1;
        }

        /* Estilo do Modal de Pilares */
        .pillars-modal-overlay {
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(15, 23, 42, 0.95);
            backdrop-filter: blur(8px);
            z-index: 9999;
            display: none; /* Escondido por padrão */
            justify-content: center;
            align-items: center;
            opacity: 0;
            transition: opacity 0.4s ease;
        }
        .pillars-modal-content {
            background: #0f172a;
            border: 1px solid rgba(110, 231, 183, 0.3);
            border-radius: 12px;
            padding: 40px;
            width: 90%;
            max-width: 1000px;
            position: relative;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            transform: translateY(20px);
            transition: transform 0.4s ease;
        }
        .pillars-modal-overlay.active {
            opacity: 1;
        }
        .pillars-modal-overlay.active .pillars-modal-content {
            transform: translateY(0);
        }
        .btn-close-modal {
            position: absolute;
            top: 20px; right: 25px;
            background: none; border: none;
            color: #64748b; font-size: 28px;
            cursor: pointer; transition: color 0.3s;
        }
        .btn-close-modal:hover { color: #f87171; }

        /* Grid dos 4 Cartões dentro do Modal */
        .pillars-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 25px;
            margin-top: 30px;
        }
        .pillar-card {
            background: rgba(30, 41, 59, 0.6);
            border-left: 4px solid;
            padding: 25px;
            border-radius: 8px;
            transition: transform 0.3s ease, background 0.3s ease;
        }
        .pillar-card:hover {
            transform: translateY(-5px);
            background: rgba(30, 41, 59, 0.9);
        }
        .pillar-card h4 { font-size: 1.3rem; margin-bottom: 12px; display: flex; align-items: center; gap: 12px; }
        .pillar-card p { font-size: 1rem; color: #cbd5e1; line-height: 1.5; margin: 0; }

        /* Cores específicas de cada Pilar */
        .p-sensor { border-color: #10b981; } .p-sensor h4 { color: #10b981; }
        .p-iot { border-color: #3b82f6; }    .p-iot h4 { color: #3b82f6; }
        .p-plat { border-color: #a855f7; }   .p-plat h4 { color: #a855f7; }
        .p-ia { border-color: #fbbf24; }     .p-ia h4 { color: #fbbf24; }
    </style>

    <div class="full-center fade-up" style="padding-top: 40px;">
        <h1 class="title-main">Contextualização</h1>
        <h2 class="subtitle" style="margin-bottom: 40px;">O ecossistema das Cidades Inteligentes e a ascensão dos Gêmeos Digitais.</h2>

        <div class="context-container">

            <div class="context-world physical-world">
                <div class="world-icon"><i class="fas fa-city"></i></div>
                <h3>Mundo Físico</h3>
                <ul class="world-list">
                    <li><i class="fas fa-car"></i> Veículos e Tráfego</li>
                    <li><i class="fas fa-traffic-light"></i> Semáforos e Vias</li>
                    <li><i class="fas fa-cloud-sun"></i> Clima e Ambiente</li>
                </ul>
            </div>

            <div class="context-cycle">
                <div class="cycle-arrow top-arrow">
                    <span class="cycle-text">Sensores IoT & Dados em Tempo Real</span>
                    <i class="fas fa-long-arrow-alt-right"></i>
                </div>

                <div class="dt-core dt-core-interactive" onclick="abrirModalPilares()">
                    <i class="fas fa-infinity"></i>
                    <span>Digital Twin</span>
                </div>

                <div class="cycle-arrow bottom-arrow">
                    <i class="fas fa-long-arrow-alt-left"></i>
                    <span class="cycle-text">Otimização, Previsão & Ação</span>
                </div>
            </div>

            <div class="context-world virtual-world">
                <div class="world-icon"><i class="fas fa-laptop-code"></i></div>
                <h3>Mundo Virtual</h3>
                <ul class="world-list">
                    <li><i class="fas fa-cubes"></i> Simulação 3D (Unity)</li>
                    <li><i class="fas fa-brain"></i> Inteligência Artificial</li>
                    <li><i class="fas fa-chart-pie"></i> Dashboards Analíticos</li>
                </ul>
            </div>

        </div>

        <div class="glass-box glass-box-border-green" style="margin-top: 50px; padding: 25px 40px; width: 100%; max-width: 1200px; text-align: center;">
            <p style="font-size: 20px; color: #cbd5e1; line-height: 1.5;">
                A transformação digital exige que os <strong>Sistemas de Transporte Inteligente (ITS)</strong> deixem de ser apenas reativos para se tornarem preditivos, utilizando <strong>Gêmeos Digitais</strong> para espelhar e gerenciar a infraestrutura urbana em tempo real.
            </p>
        </div>
    </div>

    <div id="modal-pilares" class="pillars-modal-overlay">
        <div class="pillars-modal-content">
            <button class="btn-close-modal" onclick="fecharModalPilares()"><i class="fas fa-times"></i></button>

            <h2 style="color: #f8fafc; font-size: 28px; text-align: center; margin-bottom: 10px;">Os 4 Pilares do Gêmeo Digital</h2>
            <p style="text-align: center; color: #94a3b8; font-size: 16px;">A fundação tecnológica necessária para replicar a via física no ambiente virtual.</p>

            <div class="pillars-grid">
                <div class="pillar-card p-sensor">
                    <h4><i class="fas fa-broadcast-tower"></i> 1. Sensores</h4>
                    <p>Responsáveis por capturar a realidade da via, como o volume de veículos e o clima.</p>
                </div>

                <div class="pillar-card p-iot">
                    <h4><i class="fas fa-network-wired"></i> 2. Internet das Coisas (IoT)</h4>
                    <p>Conecta esses dispositivos físicos, garantindo a troca rápida e segura dos dados.</p>
                </div>

                <div class="pillar-card p-plat">
                    <h4><i class="fas fa-server"></i> 3. Plataformas de Gerenciamento</h4>
                    <p>Fornecem o ambiente de software para integrar e visualizar essas informações.</p>
                </div>

                <div class="pillar-card p-ia">
                    <h4><i class="fas fa-brain"></i> 4. Inteligência Artificial</h4>
                    <p>Processa essa massa de dados para reconhecer padrões, prever congestionamentos e otimizar rotas.</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        function abrirModalPilares() {
            const modal = document.getElementById('modal-pilares');
            modal.style.display = 'flex';
            // Pequeno delay para a animação de fade in funcionar corretamente
            setTimeout(() => {
                modal.classList.add('active');
            }, 10);
        }

        function fecharModalPilares() {
            const modal = document.getElementById('modal-pilares');
            modal.classList.remove('active');
            // Aguarda o tempo da animação (0.4s) para dar display none
            setTimeout(() => {
                modal.style.display = 'none';
            }, 400);
        }
    </script>
</section>