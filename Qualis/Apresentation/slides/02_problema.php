<section class="slide" id="slide-2">
    <style>
        .barrier-card {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(248, 113, 113, 0.3);
            border-radius: 8px;
            padding: 15px;
            border-left: 3px solid #f87171;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }

        /* Efeito ao passar o mouse no card */
        .barrier-card:hover {
            transform: translateY(-5px);
            background: rgba(30, 41, 59, 0.9);
            border-color: rgba(248, 113, 113, 0.8);
            box-shadow: 0 10px 25px rgba(248, 113, 113, 0.2);
        }

        /* Efeito de brilho suave no ícone ao passar o mouse */
        .barrier-card i {
            transition: all 0.3s ease;
        }
        .barrier-card:hover i {
            transform: scale(1.15);
            text-shadow: 0 0 12px rgba(248, 113, 113, 0.6);
        }
    </style>

    <div class="split-layout">
        <div class="split-left">
            <h1 class="title-main fade-up" style="text-align: left; font-size: 50px;">O Problema de Pesquisa</h1>

            <div style="margin-top: 30px; display: flex; flex-direction: column; gap: 20px;">

                <div class="fade-up" style="animation-delay: 0.2s;">
                    <h3 style="color: #f87171; font-size: 1.3rem; margin-bottom: 15px; display: flex; align-items: center; gap: 10px;">
                        <i class="fas fa-exclamation-triangle"></i> As Barreiras de Escala
                    </h3>

                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">

                        <div class="barrier-card">
                            <i class="fas fa-dollar-sign" style="color: #f87171; font-size: 1.2rem; margin-bottom: 8px;"></i>
                            <h4 style="color: #e2e8f0; font-size: 1rem; margin-bottom: 5px;">Custos</h4>
                            <p style="font-size: 0.85rem; color: #94a3b8; line-height: 1.3;">Dependência de hardware físico caro.</p>
                        </div>

                        <div class="barrier-card">
                            <i class="fas fa-network-wired" style="color: #f87171; font-size: 1.2rem; margin-bottom: 8px;"></i>
                            <h4 style="color: #e2e8f0; font-size: 1rem; margin-bottom: 5px;">Legados</h4>
                            <p style="font-size: 0.85rem; color: #94a3b8; line-height: 1.3;">Falta de interoperabilidade sistêmica.</p>
                        </div>

                        <div class="barrier-card">
                            <i class="fas fa-user-shield" style="color: #f87171; font-size: 1.2rem; margin-bottom: 8px;"></i>
                            <h4 style="color: #e2e8f0; font-size: 1rem; margin-bottom: 5px;">Privacidade</h4>
                            <p style="font-size: 0.85rem; color: #94a3b8; line-height: 1.3;">Questões éticas na coleta de dados.</p>
                        </div>

                        <div class="barrier-card">
                            <i class="fas fa-wind" style="color: #f87171; font-size: 1.2rem; margin-bottom: 8px;"></i>
                            <h4 style="color: #e2e8f0; font-size: 1rem; margin-bottom: 5px;">Dinamismo</h4>
                            <p style="font-size: 0.85rem; color: #94a3b8; line-height: 1.3;">Simuladores estáticos e isolados.</p>
                        </div>

                    </div>
                </div>

                <div class="problem-card highlight-card fade-up" style="animation-delay: 0.4s; padding: 25px; margin-top: 10px;">
                    <div class="problem-icon"><i class="fas fa-bullseye" style="color: var(--neon-blue);"></i></div>
                    <div>
                        <h3 style="color: var(--neon-blue);">A Questão Central</h3>
                        <p style="font-style: italic; font-size: 1.1rem; line-height: 1.4;">
                            Como estruturar um modelo <strong>prático, de baixo custo e sustentável</strong> que supere essas quatro barreiras?
                        </p>
                    </div>
                </div>

            </div>
        </div>

        <div class="split-right">
            <div class="scanner-container">
                <div class="scanner-line"></div>
                <div class="scanner-overlay"></div>
            </div>
        </div>
    </div>
</section>