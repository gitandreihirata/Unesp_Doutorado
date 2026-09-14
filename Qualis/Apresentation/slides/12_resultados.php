<section class="slide" id="slide-12">
    <div class="full-center fade-up" style="padding-top: 30px;">
        <h1 class="title-main" style="text-align: left; width: 100%; max-width: 1300px;">Resultados Preliminares</h1>
        <p style="color: #cbd5e1; font-size: 22px; width: 100%; max-width: 1300px; text-align: left; margin-bottom: 40px;">
            Validação do fluxo <strong>End-to-End</strong>: O encanamento de dados está operacional.
        </p>

        <div class="pipeline-layout">

            <div class="pipeline-steps-container">
                <div class="pipeline-line">
                    <div class="data-particle" style="animation-delay: 0s;"></div>
                    <div class="data-particle" style="animation-delay: 1.5s;"></div>
                    <div class="data-particle" style="animation-delay: 3s;"></div>
                </div>

                <div class="pipe-step step-green">
                    <div class="pipe-node"><i class="fas fa-cubes"></i></div>
                    <div class="pipe-content">
                        <h3>1. Simulação (Unity 3D)</h3>
                        <p>Geração contínua de eventos, física e empacotamento JSON.</p>
                    </div>
                </div>

                <div class="pipe-step step-blue">
                    <div class="pipe-node"><i class="fas fa-database"></i></div>
                    <div class="pipe-content">
                        <h3>2. Persistência (MongoDB)</h3>
                        <p>Recepção via API REST em servidor remoto de alta disponibilidade.</p>
                    </div>
                </div>

                <div class="pipe-step step-purple">
                    <div class="pipe-node"><i class="fas fa-chart-pie"></i></div>
                    <div class="pipe-content">
                        <h3>3. Monitoramento</h3>
                        <p>Consulta assíncrona e visualização de KPIs no Dashboard.</p>
                    </div>
                </div>
            </div>

            <div class="pipeline-preview">
                <div class="hologram-wrapper">
                    <div class="status-badge-online">
                        <i class="fas fa-wifi"></i> SYSTEM ONLINE
                    </div>
                    <img src="https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=800&q=80" alt="Dashboard Analítico" class="hologram-image">
                    <div class="hologram-scanline"></div>
                </div>

                <div class="demo-badges">
                    <span class="demo-badge"><i class="fas fa-gamepad"></i> WebGL Cloud Arcade</span>
                    <span class="demo-badge"><i class="fas fa-globe"></i> Web Dashboard</span>
                </div>
            </div>

        </div>
    </div>
</section>