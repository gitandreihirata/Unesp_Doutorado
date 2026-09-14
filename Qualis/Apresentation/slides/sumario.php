<section class="slide" id="slide-sumario">
    <style>
        :root {
            --cor-fase1: #10b981; /* Verde (Fundamentação) */
            --cor-fase2: #fbbf24; /* Amarelo (Desenvolvimento) */
            --cor-fase3: #f87171; /* Vermelho (Resultados/Conclusão) */
        }

        /* Divisórias Verticais no Fundo */
        .phase-divider {
            position: absolute;
            top: 20px;
            bottom: 20px;
            border-right: 2px dashed rgba(255, 255, 255, 0.15);
            z-index: 0;
            pointer-events: none; /* Garante que a linha não bloqueie o clique nas bolinhas */
        }

        /* Títulos de cada Fase (Ajustado para top: 40px para descer) */
        .phase-title {
            position: absolute;
            top: 40px;
            font-size: 1.1rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            font-weight: bold;
            opacity: 0.8;
            z-index: 0;
            text-align: center;
            pointer-events: none; /* Garante que o texto não bloqueie o mouse */
        }

        /* Cores Aplicadas aos Pontos (Bolinhas) */
        .fase-1 .point-dot { background: rgba(16, 185, 129, 0.15); border-color: var(--cor-fase1); box-shadow: 0 0 15px rgba(16, 185, 129, 0.4); }
        .fase-2 .point-dot { background: rgba(251, 191, 36, 0.15); border-color: var(--cor-fase2); box-shadow: 0 0 15px rgba(251, 191, 36, 0.4); }
        .fase-3 .point-dot { background: rgba(248, 113, 113, 0.15); border-color: var(--cor-fase3); box-shadow: 0 0 15px rgba(248, 113, 113, 0.4); }

        /* Cores Aplicadas aos Cartões (Borda superior) */
        .fase-1 .point-card { border-top: 3px solid var(--cor-fase1); }
        .fase-2 .point-card { border-top: 3px solid var(--cor-fase2); }
        .fase-3 .point-card { border-top: 3px solid var(--cor-fase3); }

        /* Cor dos botões internos (Chips) seguindo a fase */
        .fase-1 .chip-btn { border-color: rgba(16, 185, 129, 0.5); color: #e2e8f0; }
        .fase-2 .chip-btn { border-color: rgba(251, 191, 36, 0.5); color: #e2e8f0; }
        .fase-3 .chip-btn { border-color: rgba(248, 113, 113, 0.5); color: #e2e8f0; }
    </style>

    <div class="full-center fade-up" style="padding-top: 40px;">
        <h1 class="title-main" style="margin-bottom: 5px;">Roteiro da Apresentação</h1>
        <p class="subtitle" style="margin-bottom: 40px;">O percurso da pesquisa: da fundamentação teórica à validação experimental.</p>

        <div class="roadmap-container">

            <div class="phase-title" style="left: 0; width: 42%; color: var(--cor-fase1);">Fase 1: Fundamentação</div>
            <div class="phase-divider" style="left: 42%;"></div>

            <div class="phase-title" style="left: 42%; width: 36%; color: var(--cor-fase2);">Fase 2: Arquitetura</div>
            <div class="phase-divider" style="left: 78%;"></div>

            <div class="phase-title" style="left: 78%; width: 22%; color: var(--cor-fase3);">Fase 3: Resultados</div>

            <svg class="road-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
                <path class="road-path" d="M 5 45 C 20 100, 30 100, 50 45 C 70 -10, 80 -10, 95 45"></path>
            </svg>

            <div class="cyber-car" id="cyber-car" style="left: 6%; top: 52%; opacity: 1; transition: all 0.8s cubic-bezier(0.25, 1, 0.5, 1);">
                <div class="car-body" id="car-body">
                    <i class="fas fa-car-side"></i>
                    <div class="headlight"></div>
                </div>
            </div>

            <div class="roadmap-grid">

                <div class="road-point fase-1" data-index="0" style="left: 6%; top: 52%;">
                    <div class="point-dot" onclick="goToSlide(2)">1</div>
                    <div class="point-card"><h4>1. Contextualização</h4><p>Digital Twins e Smart Cities.</p></div>
                </div>

                <div class="road-point fase-1" data-index="1" style="left: 15%; top: 75%;">
                    <div class="point-dot" onclick="goToSlide(3)">2</div>
                    <div class="point-card"><h4>2. O Problema</h4><p>O desafio da integração.</p></div>
                </div>

                <div class="road-point fase-1" data-index="2" style="left: 24%; top: 85%;">
                    <div class="point-dot" onclick="goToSlide(4)">3</div>
                    <div class="point-card"><h4>3. Motivação</h4><p>Barreiras Tecnológicas.</p></div>
                </div>

                <div class="road-point fase-1" data-index="3" style="left: 35%; top: 74%;">
                    <div class="point-dot" onclick="goToSlide(5)">4</div>
                    <div class="point-card">
                        <h4>4. Objetivos</h4><p>Metas e Hipótese.</p>
                        <div class="sub-nav-chips">
                            <button class="chip-btn" onclick="goToSlide(5)">Geral</button>
                            <button class="chip-btn" onclick="goToSlide(6)">Hipótese</button>
                        </div>
                    </div>
                </div>

                <div class="road-point fase-2" data-index="4" style="left: 45%; top: 52%;">
                    <div class="point-dot" onclick="goToSlide(7)">5</div>
                    <div class="point-card">
                        <h4>5. Inovação</h4><p>Estado da Arte (SUMO/CARLA).</p>
                    </div>
                </div>

                <div class="road-point fase-2" data-index="5" style="left: 55%; top: 30%;">
                    <div class="point-dot" onclick="goToSlide(8)">6</div>
                    <div class="point-card"><h4>6. A Proposta</h4><p>SmartCitySystem.</p></div>
                </div>

                <div class="road-point fase-2" data-index="6" style="left: 65%; top: 12%;">
                    <div class="point-dot" onclick="goToSlide(9)">7</div>
                    <div class="point-card">
                        <h4>7. Metodologia</h4><p>Fases de Desenvolvimento.</p>
                        <div class="sub-nav-chips">
                            <button class="chip-btn" onclick="goToSlide(9)">Geral</button>
                            <button class="chip-btn" onclick="goToSlide(10)">Fases</button>
                        </div>
                    </div>
                </div>

                <div class="road-point fase-2" data-index="7" style="left: 75%; top: 3%;">
                    <div class="point-dot" onclick="goToSlide(11)">8</div>
                    <div class="point-card"><h4>8. Arquitetura</h4><p>Modelo de Três Camadas.</p></div>
                </div>

                <div class="road-point fase-3" data-index="8" style="left: 82%; top: 10%;">
                    <div class="point-dot" onclick="goToSlide(12)">9</div>
                    <div class="point-card" style="width: 220px;">
                        <h4>9. Implementação</h4><p>Sensores e Integração Real.</p>
                        <div class="sub-nav-chips">
                            <button class="chip-btn" onclick="goToSlide(12)">Sensores</button>
                            <button class="chip-btn" onclick="goToSlide(13)">Dados Reais</button>
                        </div>
                    </div>
                </div>

                <div class="road-point fase-3" data-index="9" style="left: 88%; top: 22%;">
                    <div class="point-dot" onclick="goToSlide(14)">10</div>
                    <div class="point-card" style="width: 220px;">
                        <h4>10. Resultados</h4><p>Validação e Dashboards.</p>
                        <div class="sub-nav-chips">
                            <button class="chip-btn green" onclick="goToSlide(15)">WebGL</button>
                            <button class="chip-btn green" onclick="goToSlide(16)">JSON</button>
                            <button class="chip-btn green" onclick="goToSlide(17)">Dash</button>
                        </div>
                    </div>
                </div>

                <div class="road-point fase-3" data-index="10" style="left: 93%; top: 38%;">
                    <div class="point-dot" onclick="goToSlide(18)"><i class="fas fa-flask"></i></div>
                    <div class="point-card"><h4>11. Cenários</h4><p>Casos de Experimentação.</p></div>
                </div>

                <div class="road-point fase-3" data-index="11" style="left: 96%; top: 52%;">
                    <div class="point-dot" onclick="goToSlide(19)">12</div>
                    <div class="point-card" style="width: 220px;">
                        <h4>12. Cronograma</h4><p>Conclusões e Inteligência Artificial.</p>
                        <div class="sub-nav-chips">
                            <button class="chip-btn" onclick="goToSlide(20)">Conclusão</button>
                            <button class="chip-btn" onclick="goToSlide(21)">Trabalhos Futuros</button>
                        </div>
                    </div>
                </div>

                <div class="road-point fase-3" data-index="12" style="left: 98%; top: 68%;">
                    <div class="point-dot" onclick="goToSlide(22)"><i class="fas fa-check-circle"></i></div>
                    <div class="point-card"><h4>13. Arguição</h4><p>Banca Examinadora.</p></div>
                </div>

            </div>

        </div>
    </div>
</section>