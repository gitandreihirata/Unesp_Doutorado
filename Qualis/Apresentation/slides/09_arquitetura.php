<section class="slide" id="slide-9">
    <div class="full-center fade-up" style="padding-top: 20px;">
        <h1 class="title-main">Arquitetura SmartCitySystem</h1>
        <h2 class="subtitle" style="margin-bottom: 40px;">Sistema escalável em três camadas independentes.</h2>

        <div class="arch-layout">
            <div class="arch-text">
                <div class="arch-layer" data-target="1">
                    <h4>Camada 1: Simulação e Sensores</h4>
                    <p>Ambiente Unity 3D, IA e sensores virtuais emulados gerando dados precisos de tráfego.</p>
                    <div class="down-arrow-box"><i class="fas fa-arrow-down"></i></div>
                </div>

                <div class="arch-layer blue" data-target="2">
                    <h4>Camada 2: Backend Node.js</h4>
                    <p>API RESTful recebendo, validando e processando os dados assíncronos via HTTP POST.</p>
                    <div class="down-arrow-box"><i class="fas fa-arrow-down"></i></div>
                </div>

                <div class="arch-layer" data-target="3">
                    <h4>Camada 3: Armazenamento</h4>
                    <p>Banco de dados MongoDB (NoSQL) garantindo flexibilidade estrutural e alta velocidade.</p>
                </div>
            </div>

            <div class="arch-visual">
                <div class="iso-stack" id="iso-stack">

                    <div class="iso-layer" id="iso-1">
                        <div class="layer-title"><i class="fas fa-cubes"></i> 1. Unity 3D</div>
                        <div class="layer-content">
                            <i class="fas fa-car sim-car"></i>
                            <i class="fas fa-traffic-light"></i>
                        </div>
                    </div>

                    <div class="iso-layer" id="iso-2">
                        <div class="layer-title"><i class="fas fa-server"></i> 2. Node.js API</div>
                        <div class="layer-content">
                            <div class="api-packet">POST /sensor</div>
                            <div class="api-packet delay">POST /traffic</div>
                        </div>
                    </div>

                    <div class="iso-layer" id="iso-3">
                        <div class="layer-title"><i class="fas fa-database"></i> 3. MongoDB</div>
                        <div class="layer-content">
                            <div class="bson-doc">{ "id": 1, "status": "ok" }</div>
                        </div>
                    </div>

                </div>
            </div>
        </div>
    </div>
</section>