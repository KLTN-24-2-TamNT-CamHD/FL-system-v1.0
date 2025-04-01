import React, { useState } from 'react';
import axios from 'axios';
import { Form } from '../styles';

const InstitutionRegistration = () => {
    const [institutionName, setInstitutionName] = useState('');

    const handleRegister = (event) => {
        event.preventDefault();
        axios.post('/api/institutions/register', { name: institutionName }) // Adjusted URL
            .then(response => {
                console.log('Institution registered:', response.data);
                // Handle success (e.g., show a success message)
            })
            .catch(error => {
                console.error('Error registering institution:', error);
                // Handle error (e.g., show an error message)
            });
    };

    return (
        <div>
            <h2>Register Institution</h2>
            <Form onSubmit={handleRegister}>
                <label>Institution Name:</label>
                <input
                    type="text"
                    value={institutionName}
                    onChange={(e) => setInstitutionName(e.target.value)}
                />
                <button type="submit">Register</button>
            </Form>
        </div>
    );
};

export default InstitutionRegistration;