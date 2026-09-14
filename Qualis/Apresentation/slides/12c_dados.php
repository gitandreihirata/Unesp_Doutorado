<section class="slide" id="slide-12c">
    <div class="full-center fade-up" style="padding-top: 30px;">
        <h1 class="title-main" style="margin-bottom: 5px;">Geração e Estrutura de Dados</h1>
        <p class="subtitle" style="margin-bottom: 30px;">Documentos JSON gerados atomizadamente e enriquecidos pelo contexto real.</p>

        <div style="display: flex; gap: 40px; width: 100%; max-width: 1300px; position: relative;">

            <div class="post-anim-center">
                <div class="post-ping"></div>
                <i class="fas fa-exchange-alt"></i>
                <span>HTTP POST</span>
            </div>

            <div class="code-window live-terminal terminal-green">
                <div class="code-header">
                    <div class="browser-dots"><span></span><span></span><span></span></div>
                    <span><i class="fas fa-satellite-dish blink-icon"></i> TrafficLightData.json</span>
                </div>
                <pre class="code-content"><div class="scanline"></div><code>{
  <span class="code-key">"timestamp"</span>: <span class="code-string enriched-data">"<span id="live-time-1">2026-04-20T10:32:05.000Z</span>"</span> <i class="fas fa-clock enriched-icon" title="Dado da WorldTimeAPI"></i>,
  <span class="code-key">"tipo_sensor"</span>: <span class="code-string">"semaforo"</span>,
  <span class="code-key">"estado"</span>: <span class="code-string">"<span id="live-state">vermelho</span>"</span>,
  <span class="code-key">"localizacao"</span>: <span class="code-string">"cruzamento_01"</span>,
  <span class="code-key">"clima_atual"</span>: <span class="code-string enriched-data">"Rain"</span> <i class="fas fa-cloud-showers-heavy enriched-icon" title="Dado do OpenWeatherMap"></i>
}</code></pre>
            </div>

            <div class="code-window live-terminal terminal-blue">
                <div class="code-header">
                    <div class="browser-dots"><span></span><span></span><span></span></div>
                    <span><i class="fas fa-car-side blink-icon"></i> VehicleCountData.json</span>
                </div>
                <pre class="code-content"><div class="scanline"></div><code>{
  <span class="code-key">"timestamp"</span>: <span class="code-string enriched-data">"<span id="live-time-2">2026-04-20T10:32:05.000Z</span>"</span> <i class="fas fa-clock enriched-icon" title="Dado da WorldTimeAPI"></i>,
  <span class="code-key">"tipo_sensor"</span>: <span class="code-string">"fluxo"</span>,
  <span class="code-key">"quantidade_veiculos"</span>: <span class="code-number highlight-flash" id="live-count">5</span>,
  <span class="code-key">"tempo_medio"</span>: <span class="code-number highlight-flash" id="live-avg">2.34</span>,
  <span class="code-key">"localizacao"</span>: <span class="code-string">"via_01"</span>
}</code></pre>
            </div>

        </div>

        <div class="glass-box" style="margin-top: 40px; padding: 20px 30px; display: inline-block; border-color: rgba(110, 231, 183, 0.3);">
            <p style="font-size: 18px; color: #cbd5e1; margin: 0;">
                <i class="fas fa-shield-check" style="color: var(--neon-green); font-size: 24px; vertical-align: middle; margin-right: 10px;"></i>
                <strong>Sincronização Perfeita:</strong> Os dados marcados com ícones são anexados pelas APIs antes do disparo assíncrono para o MongoDB.
            </p>
        </div>
    </div>
</section>