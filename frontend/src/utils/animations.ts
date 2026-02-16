import type { Variants } from 'framer-motion';

// Page transition animations
export const pageVariants: Variants = {
  initial: {
    opacity: 0,
    y: 20,
  },
  animate: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.5,
      ease: 'easeOut',
    },
  },
  exit: {
    opacity: 0,
    y: -20,
    transition: {
      duration: 0.3,
    },
  },
};

// Stagger children animation
export const staggerContainer: Variants = {
  animate: {
    transition: {
      staggerChildren: 0.1,
    },
  },
};

// Card animation
export const cardVariants: Variants = {
  initial: {
    opacity: 0,
    scale: 0.9,
  },
  animate: {
    opacity: 1,
    scale: 1,
    transition: {
      duration: 0.3,
      ease: 'easeOut',
    },
  },
  hover: {
    scale: 1.02,
    y: -4,
    transition: {
      duration: 0.2,
    },
  },
};

// Slide in from left
export const slideInLeft: Variants = {
  initial: {
    x: -100,
    opacity: 0,
  },
  animate: {
    x: 0,
    opacity: 1,
    transition: {
      duration: 0.5,
      ease: 'easeOut',
    },
  },
};

// Slide in from right
export const slideInRight: Variants = {
  initial: {
    x: 100,
    opacity: 0,
  },
  animate: {
    x: 0,
    opacity: 1,
    transition: {
      duration: 0.5,
      ease: 'easeOut',
    },
  },
};

// Fade in animation
export const fadeIn: Variants = {
  initial: {
    opacity: 0,
  },
  animate: {
    opacity: 1,
    transition: {
      duration: 0.3,
    },
  },
};

// Scale in animation
export const scaleIn: Variants = {
  initial: {
    scale: 0,
    opacity: 0,
  },
  animate: {
    scale: 1,
    opacity: 1,
    transition: {
      duration: 0.3,
      ease: 'easeOut',
    },
  },
};

// Number counter animation helper
export const animateNumber = (start: number, end: number, duration: number = 1000): Promise<number[]> => {
  return new Promise((resolve) => {
    const steps = 60;
    const stepDuration = duration / steps;
    const increment = (end - start) / steps;
    const values: number[] = [];

    let current = start;
    const interval = setInterval(() => {
      current += increment;
      values.push(Math.round(current));

      if (values.length >= steps) {
        clearInterval(interval);
        resolve(values);
      }
    }, stepDuration);
  });
};

// Shake animation for errors
export const shakeAnimation = {
  x: [0, -10, 10, -10, 10, 0],
  transition: {
    duration: 0.5,
  },
};

// Pulse glow animation
export const pulseGlow: Variants = {
  animate: {
    boxShadow: [
      '0 0 20px rgba(99, 102, 241, 0.5)',
      '0 0 30px rgba(99, 102, 241, 0.8)',
      '0 0 20px rgba(99, 102, 241, 0.5)',
    ],
    transition: {
      duration: 2,
      repeat: Infinity,
      ease: 'easeInOut',
    },
  },
};

// List item animation
export const listItemVariants: Variants = {
  initial: {
    opacity: 0,
    x: -20,
  },
  animate: {
    opacity: 1,
    x: 0,
  },
  exit: {
    opacity: 0,
    x: 20,
  },
};
