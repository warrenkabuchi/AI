// ============================================================
// Purple Storm Roofing — interactions
// ============================================================

// year stamp
document.getElementById('year').textContent = new Date().getFullYear();

// ----- mobile nav toggle -----
const toggle = document.querySelector('.nav-toggle');
const links = document.querySelector('.nav-links');
toggle?.addEventListener('click', () => {
  const open = links.classList.toggle('open');
  toggle.setAttribute('aria-expanded', String(open));
});
links?.querySelectorAll('a').forEach(a => {
  a.addEventListener('click', () => {
    links.classList.remove('open');
    toggle?.setAttribute('aria-expanded', 'false');
  });
});

// ----- scroll reveal -----
const revealTargets = document.querySelectorAll(
  '.service, .step, .stat, .work-item, .review, .cities li, .faq-list details'
);
revealTargets.forEach(el => el.classList.add('reveal'));

const io = new IntersectionObserver(
  entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-in');
        io.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.15, rootMargin: '0px 0px -60px 0px' }
);
revealTargets.forEach(el => io.observe(el));

// stagger when groups appear
function staggerWithin(selector, step = 70) {
  document.querySelectorAll(selector).forEach(el => {
    [...el.children].forEach((child, i) => {
      child.style.transitionDelay = `${i * step}ms`;
    });
  });
}
staggerWithin('.service-grid', 90);
staggerWithin('.process-list', 110);
staggerWithin('.why-stats', 120);
staggerWithin('.work-grid', 80);
staggerWithin('.review-rail', 90);
staggerWithin('.cities', 30);
staggerWithin('.faq-list', 60);

// ----- form submit (Formspree-friendly with client-side success state) -----
const form = document.getElementById('estimateForm');
const success = form?.querySelector('.form-success');
const submitBtn = form?.querySelector('.btn-submit');
const submitLabel = submitBtn?.querySelector('.btn-label');

form?.addEventListener('submit', async (e) => {
  e.preventDefault();

  // basic validity check (lets browser surface field errors)
  if (!form.reportValidity()) return;

  submitBtn.disabled = true;
  if (submitLabel) submitLabel.textContent = 'Sending…';

  const action = form.getAttribute('action') || '';
  const usingPlaceholder = action.includes('YOUR_FORM_ID') || !action;

  try {
    if (!usingPlaceholder) {
      const data = new FormData(form);
      const res = await fetch(action, {
        method: 'POST',
        body: data,
        headers: { Accept: 'application/json' }
      });
      if (!res.ok) throw new Error('Submission failed');
    } else {
      // No real endpoint configured — pretend, but log so dev sees payload.
      await new Promise(r => setTimeout(r, 600));
      console.info('[estimate] payload (no endpoint configured):',
        Object.fromEntries(new FormData(form).entries()));
    }

    // Show success state
    form.querySelectorAll('.form-row, .field, .field-group, .consent, .btn-submit').forEach(el => {
      el.style.display = 'none';
    });
    success.hidden = false;
    success.scrollIntoView({ behavior: 'smooth', block: 'center' });
  } catch (err) {
    if (submitLabel) submitLabel.textContent = 'Something went wrong — try again';
    submitBtn.disabled = false;
    console.error(err);
  }
});

// ----- subtle phone formatting on input -----
const phoneInput = form?.querySelector('input[name="phone"]');
phoneInput?.addEventListener('input', (e) => {
  const digits = e.target.value.replace(/\D/g, '').slice(0, 10);
  if (digits.length < 4) e.target.value = digits;
  else if (digits.length < 7) e.target.value = `(${digits.slice(0,3)}) ${digits.slice(3)}`;
  else e.target.value = `(${digits.slice(0,3)}) ${digits.slice(3,6)}-${digits.slice(6)}`;
});
