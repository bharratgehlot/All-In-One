function toggleSemester(semesterId) {
      const semester = document.getElementById(semesterId);
      const content = semester.querySelector('.semester-content');
      const icon = semester.querySelector('.toggle-icon');

      // Close all other semesters

      document.querySelectorAll('.semester.expanded').forEach(otherSemester => {
        if (otherSemester.id !== semesterId) {
          const otherContent = otherSemester.querySelector('.semester-content');
          const otherIcon = otherSemester.querySelector('.toggle-icon');
          otherContent.style.maxHeight = '0px';
          otherSemester.classList.remove('expanded');
          otherIcon.textContent = '+';
        }
      })

      // Toggle current semesters
      if (semester.classList.contains('expanded')) {
        content.style.maxHeight = '0px';
        semester.classList.remove('expanded');
        icon.textContent = '+';
      } else {
        content.style.maxHeight = content.scrollHeight + 'px';
        semester.classList.add('expanded');
        icon.textContent = '-';
      }
    }


    function toggleSubject(subjectId) {
      const content = document.getElementById(subjectId);
      const header = content.previousElementSibling;
      const icon = header.querySelector('.subject-toggle');

    // Close all other subjects in the same semester
    const semester = content.closest('.semester');
    semester.querySelectorAll('.subject-content').forEach(otherContent => {
      if (otherContent.id !== subjectId) {
        const otherHeader = otherContent.previousElementSibling;
        const otherIcon = otherHeader.querySelector('.subject-toggle');
        otherContent.style.maxHeight = '0px';
        otherIcon.textContent = '+';
      }
    });

      // Toggle current subject
      if (content.style.maxHeight && content.style.maxHeight !== '0px') {
        content.style.maxHeight = '0px';
        icon.textContent = '+';
      } else {
        content.style.maxHeight = content.scrollHeight + 'px';
        icon.textContent = '-';
      }
    }
