<section class="slide" id="slide-12d">
    <div class="split-layout">
        <div class="split-left fade-up">
            <h1 class="title-main" style="text-align: left; font-size: 50px;">Dashboard & Armazenamento</h1>
            <p style="font-size: 20px; color: #94a3b8; line-height: 1.6; margin-bottom: 30px;">
                Os dados JSON recebidos são persistidos com segurança no <strong>MongoDB em nuvem</strong>. O Dashboard Web consulta essas coleções para gerar inteligência acionável.
            </p>

            <div style="display: flex; flex-direction: column; gap: 15px;">
                <div class="kpi-card">
                    <div class="kpi-icon"><i class="fas fa-car-side"></i></div>
                    <div class="kpi-info">
                        <h5>Contagem Horária</h5>
                        <p>Veículos por via / hora</p>
                    </div>
                </div>

                <div class="kpi-card">
                    <div class="kpi-icon blue"><i class="fas fa-stopwatch"></i></div>
                    <div class="kpi-info">
                        <h5>Tempo Médio</h5>
                        <p>Ritmo de passagem (segundos)</p>
                    </div>
                </div>

                <div class="kpi-card">
                    <div class="kpi-icon green"><i class="fas fa-traffic-light"></i></div>
                    <div class="kpi-info">
                        <h5>Frequência Semafórica</h5>
                        <p>Ciclo de estados operacionais</p>
                    </div>
                </div>
            </div>
        </div>

        <div class="split-right" style="display: flex; justify-content: center; align-items: center; padding-right: 50px;">
            <div class="dashboard-preview fade-up" style="animation-delay: 0.3s;">
                <img src="img/img_dashboard.png" alt="Dashboard SmartCity">
                <div class="dashboard-glow"></div>
            </div>
        </div>
    </div>
</section>