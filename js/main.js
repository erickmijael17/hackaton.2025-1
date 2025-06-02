// Lógica simple para simular login
const loginForm = document.getElementById('loginForm');
if (loginForm) {
  loginForm.addEventListener('submit', function(e) {
    e.preventDefault();
    // Aquí podrías validar usuario/contraseña
    window.location.href = 'dashboard.html';
  });
} 