<section class="slide" id="slide-10">
    <div class="split-layout">
        <div class="split-left fade-up">
            <h1 class="title-main" style="text-align: left; font-size: 50px;">Sensores Virtuais</h1>
            <p style="font-size: 20px; color: #94a3b8; margin-bottom: 30px;">Emulação de componentes físicos utilizando a física da Unity 3D.</p>

            <div class="sensor-list">

                <div class="glass-box sensor-card active-card" data-target="semaforo">
                    <h4 style="color: var(--neon-green);"><i class="fas fa-traffic-light"></i> Semáforo Inteligente</h4>
                    <ul>
                        <li>Ciclo programável (verde, amarelo, vermelho)</li>
                        <li>Feedback visual com mudança de material no 3D</li>
                        <li>JSON Payload: timestamp, estado e local</li>
                    </ul>
                </div>

                <div class="glass-box sensor-card" data-target="contador">
                    <h4 style="color: var(--neon-blue);"><i class="fas fa-car-side"></i> Contagem de Veículos</h4>
                    <ul>
                        <li>Área de detecção invisível na via (Collider Trigger)</li>
                        <li>Agregação por intervalo de tempo estipulado</li>
                        <li>JSON Payload: quantidade, tempo médio e local</li>
                    </ul>
                </div>

            </div>
        </div>

        <div class="split-right" style="display: flex; justify-content: center; align-items: center;">

            <div id="visual-semaforo" class="sensor-visual active">
                <div class="traffic-container">
                    <div class="traffic-light" id="interactive-light">
                        <div class="bulb red" id="light-red"></div>
                        <div class="bulb yellow" id="light-yellow"></div>
                        <div class="bulb green active" id="light-green"></div>
                    </div>
                </div>
            </div>

            <div id="visual-contador" class="sensor-visual">
                <div class="road-container">
                    <div class="digital-counter">FLUXO: <span id="car-count">00</span></div>

                    <div class="trigger-zone" id="trigger-box">
                        <span>TRIGGER<br>ZONE</span>
                    </div>

                    <div id="highway">
                        <div class="road-dashed-line"></div>
                    </div>
                </div>
            </div>

        </div>
    </div>
</section>