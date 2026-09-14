document.addEventListener('DOMContentLoaded', () => {
    const slides = document.querySelectorAll('.slide');
    const btnPrev = document.getElementById('btn-prev');
    const btnNext = document.getElementById('btn-next');
    const progressBar = document.getElementById('progress-bar');
    let currentSlide = 0;
    const totalSlides = slides.length;

    function updateSlide(index) {
        if (totalSlides === 0) return;
        slides.forEach(slide => slide.classList.remove('active'));

        if (index < 0) currentSlide = 0;
        else if (index >= totalSlides) currentSlide = totalSlides - 1;
        else currentSlide = index;

        slides[currentSlide].classList.add('active');

        const progress = ((currentSlide + 1) / totalSlides) * 100;
        progressBar.style.width = progress + '%';
    }

    // Função global acessível por outros scripts
    window.goToSlide = function(index) {
        if (typeof updateSlide === 'function') {
            updateSlide(index);
        }
    };

    // Botão Central de Roteiro
    const btnSummary = document.getElementById('btn-summary');
    if (btnSummary) {
        btnSummary.addEventListener('click', () => goToSlide(1));
    }

    if(btnPrev) btnPrev.addEventListener('click', () => updateSlide(currentSlide - 1));
    if(btnNext) btnNext.addEventListener('click', () => updateSlide(currentSlide + 1));

    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowRight' || e.key === ' ') {
            updateSlide(currentSlide + 1);
        } else if (e.key === 'ArrowLeft') {
            updateSlide(currentSlide - 1);
        }
    });

    updateSlide(0);

    // Scaler
    const presentationArea = document.getElementById('presentation-area');
    function scalePresentation() {
        const scale = Math.min(window.innerWidth / 1920, window.innerHeight / 1080);
        presentationArea.style.transform = `scale(${scale})`;
    }
    window.addEventListener('resize', scalePresentation);
    scalePresentation();
});