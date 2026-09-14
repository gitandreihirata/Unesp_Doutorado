document.addEventListener('DOMContentLoaded', () => {

    // =========================================================
    // LÓGICA DO SLIDE 09: ARQUITETURA 3D INTERATIVA
    // =========================================================
    const archLayers = document.querySelectorAll('.arch-layer');
    // Correção: Agora ele busca pela CLASSE ou pelo ID garantindo que encontre
    const isoStack = document.querySelector('.iso-stack');

    if (archLayers.length > 0 && isoStack) {
        archLayers.forEach(layer => {
            layer.addEventListener('mouseenter', function() {
                const targetId = this.getAttribute('data-target');
                const targetIso = document.getElementById('iso-' + targetId);

                isoStack.classList.add('is-hovered');
                if (targetIso) targetIso.classList.add('highlight');
            });

            layer.addEventListener('mouseleave', function() {
                const targetId = this.getAttribute('data-target');
                const targetIso = document.getElementById('iso-' + targetId);

                isoStack.classList.remove('is-hovered');
                if (targetIso) targetIso.classList.remove('highlight');
            });
        });
    }

    // =========================================================
    // LÓGICA DO TERMINAL DE APIS (SLIDE 11)
    // =========================================================
    const apiCards = document.querySelectorAll('.api-card-interactive');
    const terminalCode = document.getElementById('terminal-code');

    const codeData = {
        weather: `
<span style="color:var(--neon-blue); font-weight:bold;">[HTTP GET]</span> https://api.openweathermap.org/data/2.5/weather?q=SaoPaulo
<span style="color:var(--neon-green);">Status: 200 OK</span> | Content-Type: application/json | Latency: 42ms

{
  <span style="color:#93c5fd;">"coord"</span>: { <span style="color:#93c5fd;">"lon"</span>: <span style="color:#fca5a5;">-46.63</span>, <span style="color:#93c5fd;">"lat"</span>: <span style="color:#fca5a5;">-23.54</span> },
  <span style="color:#93c5fd;">"weather"</span>: [{"main": <span style="color:#86efac;">"Rain"</span>, "description": <span style="color:#86efac;">"moderate rain"</span>}],
  <span style="color:#93c5fd;">"main"</span>: {"temp": <span style="color:#fca5a5;">22.5</span>, "humidity": <span style="color:#fca5a5;">88</span>},
  <span style="color:#93c5fd;">"wind"</span>: {"speed": <span style="color:#fca5a5;">5.1</span>}
}

<span style="color:#64748b;">// Ação na Unity: String "Rain" interceptada. 
// Ativando ParticleSystem (Chuva) e reduzindo atrito em 15%.</span>`,

        time: `
<span style="color:var(--neon-green); font-weight:bold;">[HTTP GET]</span> http://worldtimeapi.org/api/timezone/America/Sao_Paulo
<span style="color:var(--neon-green);">Status: 200 OK</span> | Content-Type: application/json | Latency: 18ms

{
  <span style="color:#93c5fd;">"abbreviation"</span>: <span style="color:#86efac;">"-03"</span>,
  <span style="color:#93c5fd;">"datetime"</span>: <span style="color:#86efac;">"2026-04-20T10:30:00.123-03:00"</span>,
  <span style="color:#93c5fd;">"timezone"</span>: <span style="color:#86efac;">"America/Sao_Paulo"</span>,
  <span style="color:#93c5fd;">"unixtime"</span>: <span style="color:#fca5a5;">1776691800</span>
}

<span style="color:#64748b;">// Ação na Unity: Data ISO-8601 e UNIX Time interceptados.
// Configurando timestamp global do payload.</span>`
    };

    if (apiCards.length > 0 && terminalCode) {
        apiCards.forEach(card => {
            card.addEventListener('mouseenter', () => {
                const apiType = card.getAttribute('data-api');
                apiCards.forEach(c => c.style.opacity = '0.5');
                card.style.opacity = '1';
                terminalCode.innerHTML = codeData[apiType];
            });

            card.addEventListener('mouseleave', () => {
                apiCards.forEach(c => c.style.opacity = '1');
                terminalCode.innerHTML = `
<span style="color:#64748b;">// Aguardando disparo de rotina HTTP...</span>
<span style="color:#64748b;">// Passe o mouse sobre as APIs acima para interceptar o Payload.</span>`;
            });
        });
    }

    // =========================================================
    // LÓGICA DO SLIDE 12C: JSON LIVE SIMULATOR
    // =========================================================
    const time1 = document.getElementById('live-time-1');
    const time2 = document.getElementById('live-time-2');
    const state = document.getElementById('live-state');
    const count = document.getElementById('live-count');
    const avg = document.getElementById('live-avg');
    const slide12c = document.getElementById('slide-12c');

    if (time1 && time2 && slide12c) {
        const states = ["verde", "amarelo", "vermelho"];
        let stateIndex = 0;

        setInterval(() => {
            if (slide12c.classList.contains('active')) {
                const now = new Date().toISOString();
                time1.innerText = now;
                time2.innerText = now;

                stateIndex = (stateIndex + 1) % states.length;
                state.innerText = states[stateIndex];

                const newCount = Math.floor(Math.random() * 8) + 1;
                const newAvg = (Math.random() * 3 + 1.5).toFixed(2);

                count.innerText = newCount;
                avg.innerText = newAvg;

                count.classList.add('flash-active-blue');
                avg.classList.add('flash-active-blue');
                state.classList.add('flash-active-green');

                setTimeout(() => {
                    count.classList.remove('flash-active-blue');
                    avg.classList.remove('flash-active-blue');
                    state.classList.remove('flash-active-green');
                }, 300);
            }
        }, 2500);
    }
});