document.addEventListener('DOMContentLoaded', () => {
    const btnOpenTable = document.getElementById('btn-open-table');
    const btnCloseTable = document.getElementById('btn-close-table');
    const modalCompare = document.getElementById('modal-compare');

    if (btnOpenTable && btnCloseTable && modalCompare) {
        btnOpenTable.addEventListener('click', () => {
            modalCompare.classList.add('show');
        });

        btnCloseTable.addEventListener('click', () => {
            modalCompare.classList.remove('show');
        });

        modalCompare.addEventListener('click', (e) => {
            if (e.target === modalCompare) {
                modalCompare.classList.remove('show');
            }
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && modalCompare.classList.contains('show')) {
                modalCompare.classList.remove('show');
            }
        });
    }
});